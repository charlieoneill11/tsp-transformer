#!/usr/bin/env python3
"""
generate_tsp_instances.py

Script to generate random TSP instances with integer coordinates for given sizes.
Optionally solves each instance using OR-Tools for a (near-)optimal route.

Requirements:
  - pip install ortools

Example usage:
  python generate_tsp_instances.py \
    --sizes 5 10 15 20 \
    --num_instances 10 \
    --output_dir data/tsp \
    --solve \
    --seed 42
"""

import os
import json
import argparse
import random
import math
from typing import List, Dict, Any, Tuple

# Check OR-Tools availability
try:
    from ortools.constraint_solver import pywrapcp, routing_enums_pb2
    ORTOOLS_AVAILABLE = True
except ImportError:
    ORTOOLS_AVAILABLE = False
    print("Warning: OR-Tools not installed. TSP solutions won't be generated.")


def generate_random_integer_coordinates(n_nodes: int, seed: int = None) -> List[Tuple[int, int]]:
    """
    Generate a list of random integer (x, y) coordinates in [0, 100].
    :param n_nodes: Number of nodes (cities).
    :param seed: Optional random seed.
    :return: List of (x, y) tuples (integers in [0, 100]).
    """
    if seed is not None:
        random.seed(seed)

    coords = [(random.randint(0, 100), random.randint(0, 100)) for _ in range(n_nodes)]
    return coords


def compute_euclidean_distance_matrix(coords: List[Tuple[int, int]]) -> List[List[float]]:
    """
    Compute the full NxN Euclidean distance matrix for integer coordinates.

    :param coords: List of (x, y) tuples for each city.
    :return: NxN matrix of pairwise distances.
    """
    n = len(coords)
    dist_matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                dist_matrix[i][j] = 0.0
            else:
                (x1, y1), (x2, y2) = coords[i], coords[j]
                dist_matrix[i][j] = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    return dist_matrix


def solve_tsp_ortools(distance_matrix: List[List[float]]) -> Tuple[List[int], float]:
    """
    Solve a TSP instance using OR-Tools. 
    Returns the best route found and the total route cost (in floating-point, not scaled).

    :param distance_matrix: NxN matrix of pairwise distances.
    :return: (route, route_cost)
    """
    if not ORTOOLS_AVAILABLE:
        raise RuntimeError("OR-Tools is not installed. Cannot solve TSP.")

    manager = pywrapcp.RoutingIndexManager(len(distance_matrix), 1, 0)
    routing = pywrapcp.RoutingModel(manager)

    # Create the distance callback.
    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        # OR-Tools requires integer distances, so scale up
        return int(distance_matrix[from_node][to_node] * 1000)

    transit_callback_idx = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_idx)

    # Setting first solution heuristic
    search_params = pywrapcp.DefaultRoutingSearchParameters()
    search_params.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )

    # Solve the problem
    solution = routing.SolveWithParameters(search_params)
    if solution:
        # Extract the route
        route = []
        index = routing.Start(0)
        route_cost = 0.0
        while not routing.IsEnd(index):
            route.append(manager.IndexToNode(index))
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            if not routing.IsEnd(index):
                route_cost += distance_matrix[route[-1]][manager.IndexToNode(index)]
        return route, route_cost
    else:
        return [], float('inf')


def generate_tsp_instances(
    sizes: List[int],
    num_instances: int,
    output_dir: str,
    solve: bool = False,
    seed: int = 42
) -> None:
    """
    Generate random TSP data for given sizes with integer coordinates. Optionally solve each instance.

    :param sizes: List of TSP sizes, e.g. [5, 10, 15, 20].
    :param num_instances: How many instances to create per size.
    :param output_dir: Directory to save the generated data files.
    :param solve: Whether to run a TSP solver (OR-Tools) to get routes.
    :param seed: Random seed for reproducibility.
    """
    os.makedirs(output_dir, exist_ok=True)

    for tsp_size in sizes:
        data_for_size = []
        print(f"[INFO] Generating {num_instances} instances for TSP size {tsp_size}...")

        for i in range(num_instances):
            # Use seed + i for reproducible but distinct coordinates
            coords = generate_random_integer_coordinates(n_nodes=tsp_size, seed=seed + i)
            distance_matrix = compute_euclidean_distance_matrix(coords)

            instance_dict = {
                "n_nodes": tsp_size,
                "coords": coords,
                "distance_matrix": distance_matrix,
                "optimal_route": None,
                "optimal_cost": None
            }

            if solve and ORTOOLS_AVAILABLE:
                route, route_cost = solve_tsp_ortools(distance_matrix)
                instance_dict["optimal_route"] = route
                # Round or keep as float
                instance_dict["optimal_cost"] = round(route_cost, 4)

            data_for_size.append(instance_dict)

        # Save to JSON
        output_path = os.path.join(output_dir, f"tsp_{tsp_size}nodes.json")
        with open(output_path, "w") as f:
            json.dump(data_for_size, f, indent=2)

        print(f"[INFO] Saved {num_instances} TSP instances of size {tsp_size} to {output_path}")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Generate random TSP instances (integer coords) and optionally solve with OR-Tools."
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        default=[5, 10, 15, 20],
        help="List of TSP sizes to generate (e.g. --sizes 5 10 15 20)."
    )
    parser.add_argument(
        "--num_instances",
        type=int,
        default=100,
        help="Number of instances per size to generate."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="tsp",
        help="Directory where data files will be saved."
    )
    parser.add_argument(
        "--solve",
        action="store_true",
        help="If provided, attempt to solve each instance with OR-Tools (must be installed)."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility."
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    generate_tsp_instances(
        sizes=args.sizes,
        num_instances=args.num_instances,
        output_dir=args.output_dir,
        solve=args.solve,
        seed=args.seed
    )


if __name__ == "__main__":
    main()