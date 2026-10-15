from collections import deque

import numpy as np

from .coding import Basis


def shortest_path(adjacency, source, destination, banned=None):
    queue, previous = deque([source]), {source: None}
    while queue:
        node = queue.popleft()
        if node == destination:
            route = []
            while node is not None:
                route.append(node)
                node = previous[node]
            return route[::-1]
        for other in np.flatnonzero(adjacency[node]):
            if (node, int(other)) == banned or other in previous:
                continue
            previous[int(other)] = node
            queue.append(int(other))
    return []


def channel_parameters(loss, burst, q_good=0.98, q_bad=0.08):
    bad_fraction = ((1 - q_good) - loss) / (q_bad - q_good)
    p_bg = 1 / burst
    p_gb = bad_fraction * p_bg / (1 - bad_fraction)
    if not 0 < bad_fraction < 1 or p_gb > 1 or p_bg > 1:
        raise ValueError("Requested mean loss and burst length are incompatible")
    return p_gb, p_bg, q_good, q_bad, bad_fraction


def generate(config, seed):
    rng = np.random.default_rng(seed)
    position = rng.uniform(0, config["area"], (config["nodes"], 2))
    distance = np.linalg.norm(position[:, None] - position[None], axis=-1)
    adjacency = (distance <= config["range"]) & (distance > 0)
    edges = np.argwhere(adjacency)
    if not len(edges):
        raise ValueError("Topology has no usable links")
    p_gb, p_bg, qg, qb, bad_fraction = channel_parameters(config["loss"], config["burst"])
    state = rng.random(len(edges)) >= bad_fraction
    states, uniforms, feedback = [], [], []
    for _ in range(config["slots"]):
        states.append(state.copy())
        uniforms.append(rng.random((config["capacity"], len(edges))))
        feedback.append(rng.random((config["capacity"], len(edges))))
        draw = rng.random(len(edges))
        state = np.where(state, draw >= p_gb, draw < p_bg)
    messages = []
    flows = [rng.choice(config["nodes"], 2, replace=False).tolist() for _ in range(config["flows"])]
    for arrival in range(0, config["slots"] - 100, config["arrival_period"]):
        for source, destination in flows:
            critical = bool(rng.random() < config["critical_fraction"])
            messages.append((source, destination, arrival, 32 if critical else 48, 60 if critical else 100, critical))
    return {"position": position, "edges": edges, "states": np.asarray(states),
            "uniforms": np.asarray(uniforms, dtype=np.float32), "feedback": np.asarray(feedback, dtype=np.float32),
            "messages": np.asarray(messages, dtype=int), "seed": seed}


def miss_risk(posterior, remaining, slack, active, config, rng, iid=False):
    if remaining <= 0:
        return 0.0
    if slack <= 0 or not len(posterior):
        return 1.0
    p_gb, p_bg, qg, qb, _ = channel_parameters(config["loss"], config["burst"])
    states = rng.random((config["risk_samples"], len(posterior))) < posterior[None]
    delivered = np.zeros(config["risk_samples"], int)
    credit = 0.0
    for _ in range(slack):
        credit += config["capacity"] / (max(1, active) * len(posterior))
        attempts = int(credit)
        credit -= attempts
        for _ in range(attempts):
            success = rng.random(states.shape) < np.where(states, qg, qb)
            delivered += success.all(1)
        draw = rng.random(states.shape)
        if iid:
            states = draw >= channel_parameters(config["loss"], config["burst"])[4]
        else:
            states = np.where(states, draw >= p_gb, draw < p_bg)
    return float(np.mean(delivered < remaining))


def run(trace, config, method, feedback_loss, no_risk=False, no_priority=False, no_innovation=False):
    seed = int(trace["seed"])
    coding_rng = np.random.default_rng(seed + 10000)
    risk_rng = np.random.default_rng(seed + 20000)
    edges = [tuple(map(int, e)) for e in trace["edges"]]
    lookup = {edge: i for i, edge in enumerate(edges)}
    adjacency = np.zeros((config["nodes"], config["nodes"]), bool)
    for source, destination in edges:
        adjacency[source, destination] = True
    p_gb, p_bg, qg, qb, bad_fraction = channel_parameters(config["loss"], config["burst"])
    posterior = np.full(len(edges), 1 - bad_fraction)
    messages = []
    for identifier, (source, destination, arrival, k, lifetime, critical) in enumerate(trace["messages"]):
        route = shortest_path(adjacency, int(source), int(destination))
        routes = [route] if route else []
        if method == "replication" and len(route) > 1:
            second = shortest_path(adjacency, int(source), int(destination), (route[0], route[1]))
            if second:
                routes.append(second)
        payload = coding_rng.integers(0, 256, (k, config["payload_bytes"]), dtype=np.uint8)
        basis = {int(source): Basis(int(k), config["payload_bytes"])}
        systematic = np.concatenate([np.eye(k, dtype=np.uint8), payload], 1)
        for vector in systematic:
            basis[int(source)].add(vector)
        messages.append({"id": identifier, "routes": routes, "source": int(source), "destination": int(destination),
                         "arrival": int(arrival), "deadline": int(arrival + lifetime), "k": int(k),
                         "weight": 2 if critical and not no_priority else 1, "critical": bool(critical),
                         "payload": payload, "basis": basis, "sent": {}, "acked": {}, "systematic": systematic,
                         "injected": 0, "known_rank": 0, "budget": 0, "credit": 0.0,
                         "delivered": False, "decoded_at": None, "risk": 1.0, "turn": 0})
    tx_bytes, feedback_bytes, decode_operations, estimation = 0, 0, 0, []
    decisions = []
    for t in range(config["slots"]):
        active = [m for m in messages if m["arrival"] <= t < m["deadline"] and not m["delivered"] and m["routes"]]
        for m in active:
            route = m["routes"][0]
            pi = np.array([posterior[lookup[e]] for e in zip(route, route[1:])])
            if method == "deadline":
                pi[:] = 1 - bad_fraction
            risk = miss_risk(pi, m["k"] - m["known_rank"], m["deadline"] - t, len(active), config, risk_rng, method == "deadline")
            m["risk"] = risk
            request = config["gain_risk"] * max(0, risk - config["delta"] / m["weight"]) if not no_risk else 0
            request += config["gain_demand"] * (m["k"] - m["known_rank"]) / (m["deadline"] - t + 1)
            if method in {"static", "fec", "rlnc"}:
                request = config["static_rate"]
            m["credit"] += np.clip(request, 0, config["max_repair"])
            m["budget"] = int(m["credit"])
            m["credit"] -= m["budget"]
            if m["id"] == 0:
                decisions.append({"slot": t, "risk": risk, "rank_deficit": m["k"] - m["known_rank"], "repair_request": m["budget"]})
        for mini in range(config["capacity"]):
            active.sort(key=lambda m: (-m["weight"] * (m["risk"] + (m["k"] - m["known_rank"]) / max(1, m["deadline"] - t)), m["id"]))
            if active:
                rotation = (t * config["capacity"] + mini) % len(active)
                active = active[rotation:] + active[:rotation]
            chosen = None
            for m in active:
                candidates = []
                for route in m["routes"]:
                    candidates.extend(zip(route, route[1:]))
                for j in range(len(candidates)):
                    edge = candidates[(m["turn"] + j) % len(candidates)]
                    sender, receiver = edge
                    if sender not in m["basis"] or not m["basis"][sender].rank:
                        continue
                    at_source = sender == m["source"]
                    limit = int(np.ceil(m["k"] * (1.3 if method == "fec" else 1.6)))
                    if at_source and method in {"fec", "static"} and m["injected"] >= limit:
                        continue
                    if at_source and m["injected"] >= m["k"] and m["budget"] <= 0 and method not in {"arq", "replication", "oracle"}:
                        continue
                    if method in {"arq", "replication"}:
                        acked = m["acked"].setdefault(edge, set())
                        available = [(p, r) for p, r in m["basis"][sender].rows.items() if p not in acked]
                        if not available:
                            continue
                        pivot, vector = available[0]
                    elif at_source and m["injected"] < m["k"]:
                        pivot, vector = m["injected"], m["systematic"][m["injected"]]
                    else:
                        pivot, vector = None, m["basis"][sender].recode(coding_rng)
                    sent = m["sent"].setdefault(edge, Basis(m["k"]))
                    novel = sent.add(vector[:m["k"]])
                    if method == "full" and not no_innovation and not novel and m["risk"] <= config["delta"] / m["weight"]:
                        continue
                    m["turn"] = (m["turn"] + j + 1) % len(candidates)
                    chosen = m, edge, pivot, vector
                    break
                if chosen:
                    break
            if chosen is None:
                continue
            m, edge, pivot, vector = chosen
            sender, receiver = edge
            index = lookup[edge]
            good = bool(trace["states"][t, index])
            estimation.append(int((posterior[index] >= 0.5) == good))
            success = method == "oracle" or trace["uniforms"][t, mini, index] < (qg if good else qb)
            acknowledged = bool(success and trace["feedback"][t, mini, index] >= feedback_loss)
            q_good_ack, q_bad_ack = qg * (1 - feedback_loss), qb * (1 - feedback_loss)
            a = q_good_ack if acknowledged else 1 - q_good_ack
            b = q_bad_ack if acknowledged else 1 - q_bad_ack
            posterior[index] = a * posterior[index] / max(a * posterior[index] + b * (1 - posterior[index]), 1e-12)
            header = 0 if method in {"arq", "replication"} else m["k"]
            tx_bytes += config["payload_bytes"] + header
            if sender == m["source"]:
                m["injected"] += 1
                if m["injected"] > m["k"]:
                    m["budget"] -= 1
            if success:
                feedback_bytes += config["ack_bytes"]
                destination_basis = m["basis"].setdefault(receiver, Basis(m["k"], config["payload_bytes"]))
                before = destination_basis.operations
                destination_basis.add(vector)
                decode_operations += destination_basis.operations - before
                if acknowledged and pivot is not None:
                    m["acked"].setdefault(edge, set()).add(pivot)
                if receiver == m["destination"]:
                    if acknowledged:
                        m["known_rank"] = destination_basis.rank
                    if destination_basis.rank == m["k"]:
                        decoded = destination_basis.decode()
                        if not np.array_equal(decoded, m["payload"]):
                            raise ArithmeticError("Finite-field decoding disagrees with the source payload")
                        m["delivered"], m["decoded_at"] = True, t
                        active = [item for item in active if not item["delivered"]]
        posterior = posterior * (1 - p_gb) + (1 - posterior) * p_bg
    successful = [m for m in messages if m["delivered"]]
    critical = [m for m in messages if m["critical"]]
    ordinary = [m for m in messages if not m["critical"]]
    source_payload = sum(m["k"] * config["payload_bytes"] for m in successful)
    total = tx_bytes + feedback_bytes
    operations_scale = sum(m["k"] ** 2 * (m["k"] + config["payload_bytes"]) for m in successful)
    class_scores = [(2, critical), (1, ordinary)]
    weighted_delivery = sum(w * np.mean([m["delivered"] for m in group]) for w, group in class_scores if group)
    total_class_weight = sum(w for w, group in class_scores if group)
    metrics = {"weighted_deadline_delivery": float(weighted_delivery / total_class_weight) if total_class_weight else None,
               "critical_miss_rate": float(np.mean([not m["delivered"] for m in critical])) if critical else None,
               "ordinary_miss_rate": float(np.mean([not m["delivered"] for m in ordinary])) if ordinary else None,
               "extra_symbols_per_delivery": (total - source_payload) / config["payload_bytes"] / len(successful) if successful else None,
               "energy_j_per_delivery": (total * config["energy_j_per_byte"] + decode_operations * config["energy_j_per_operation"]) / len(successful) if successful else None,
               "normalized_decode_cost": decode_operations / operations_scale if operations_scale else None,
               "burst_state_accuracy": float(np.mean(estimation)) if estimation else None,
               "transmitted_bytes": tx_bytes, "feedback_bytes": feedback_bytes,
               "field_operations": decode_operations, "offered": len(messages), "delivered": len(successful)}
    messages_out = [{key: m[key] for key in ["id", "arrival", "deadline", "k", "critical", "delivered", "decoded_at"]} for m in messages]
    return metrics, decisions, messages_out
