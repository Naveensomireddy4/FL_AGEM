import time
from threading import Thread, Lock
import random
import numpy as np
from collections import defaultdict
from flcore.clients.clientavg import clientAVG
from flcore.servers.serverbase import Server
from .helper import *

class FedAvg(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        self.temp = 1
        self.temp_dropout_ratio = 0
        self.count = 0
        self.total_len = 0
        self.d = 100000
        self.k = 100000  # Example value for k; set this to your desired number of clients
        self.client_times = defaultdict(list)
        self.lock = Lock()
        self.aggregated_client_count = 0
        self.has_aggregated_first_k_clients = False

        # Initialize round_dropout_clients to track dropout clients for each round
        self.round_dropout_clients = {}

        # Select slow clients
        self.set_slow_clients()
        self.set_clients(clientAVG)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        self.Budget = []

    def set_random_seed(self, round_number):
        """Set the random seed based on the round number."""
        seed = 42 + round_number  # Example fixed seed offset; adjust as needed
        random.seed(seed)
        np.random.seed(seed)

    def select_clients(self):
        """Select a fraction of clients based on join_ratio and dropout_rate."""
        num_selected_clients = max(1, int(self.join_ratio * self.num_clients))
        available_clients = random.sample(self.clients, num_selected_clients)

        selected_clients = []
        dropped_clients = []
        dropout_rate = 1  # Adjust dropout rate as needed

        # Set random seed for current round
        self.set_random_seed(self.current_round)

        threshold_val = dropout_rate * self.temp * 20
        self.temp_dropout_ratio = dropout_rate * self.temp

        for i, client in enumerate(available_clients):
            if i > threshold_val:
                selected_clients.append(client)
            else:
                dropped_clients.append(client)

        if len(selected_clients) == 0:
            selected_clients = available_clients
            dropped_clients = []
            self.temp_dropout_ratio = 0

        # Store dropout clients for the current round
        self.round_dropout_clients[self.current_round] = [client.id for client in dropped_clients]

        print("Dropped clients for round", self.current_round, ":", self.round_dropout_clients[self.current_round])
        selected_clients.sort(key=lambda client: client.id)
        dropped_clients.sort(key=lambda client: client.id)
        available_clients.sort(key=lambda client: client.id)
        return selected_clients, dropped_clients, available_clients

    def ordered_dict_to_array(self, ordered_dict):
        return np.concatenate([value.flatten() for value in ordered_dict.values()])

    def cosine_similarity(self, vec1, vec2):
        vec1_array = self.ordered_dict_to_array(vec1)
        vec2_array = self.ordered_dict_to_array(vec2)
        dot_product = np.dot(vec1_array, vec2_array)
        norm_vec1 = np.linalg.norm(vec1_array)
        norm_vec2 = np.linalg.norm(vec2_array)
        return dot_product / (norm_vec1 * norm_vec2)

    def find_nearest_trained_client(self, dropped_client, trained_clients):
        max_similarity = -1
        nearest_client = None
        dropped_client_weights = dropped_client.get_weights()

        for client in trained_clients:
            trained_client_weights = client.get_weights()
            similarity = self.cosine_similarity(dropped_client_weights, trained_client_weights)
            if similarity > max_similarity:
                max_similarity = similarity
                nearest_client = client
        return nearest_client

    def finding_nearest_clients_to_dropped_client(self, dropped_clients):
        dropped_clients_nearest_clients = []
        for dropped_client in dropped_clients:
            nearest_client = self.find_nearest_trained_client(dropped_client, self.selected_clients)
            if nearest_client:
                dropped_clients_nearest_clients.append(nearest_client)
        return dropped_clients_nearest_clients

    def train(self):
        for i in range(self.global_rounds + 1):
            s_t = time.time()
            self.current_round = i  # Track current round
            self.selected_clients, dropped_clients, _ = self.select_clients()
            self.dropped_clients_nearest_clients = self.finding_nearest_clients_to_dropped_client(dropped_clients)
            self.agg_clients = []  # all the clients which are ready for aggregation will be sent into this list
            self.send_models()

            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.evaluate()

            print("Clients:", len(self.selected_clients))

            # Parallel training using threads
            self.aggregated_client_count = 0
            self.has_aggregated_first_k_clients = False
            self.total_len = len(self.selected_clients)

            threads = []
            for client in self.selected_clients:
                thread = Thread(target=self.train_and_aggregate_client, args=(client,))
                threads.append(thread)
                thread.start()

            for thread in threads:
                thread.join()

            self.receive_models(req_clients=self.agg_clients)
            self.aggregate_parameters()

            # Calculate and print percentage difference between client and global weights
            for client in self.selected_clients:
                weights = client.get_weights()
                global_weights = self.get_global_weights()
                perc_diffs = calculate_percentage_difference(weights, global_weights)
                print("Client:", client.id, "Percentage Difference:", (1 - perc_diffs) * 100)

            self.Budget.append(time.time() - s_t)
            print('-' * 25, 'Time cost', '-' * 25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:]) / len(self.Budget[1:]))

        self.save_results()
        self.save_global_model()

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientAVG)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()

    def train_and_aggregate_client(self, client):
        """Train a single client and aggregate its model if k clients have completed training."""
        st = time.time()
        client.train()
        
        train_time = time.time() - st
        train_accuracy = client.get_train_accuracy()
        print(f"Training client: {client.id}, Time Cost: {train_time}, Train Accuracy: {train_accuracy:.2f}%")

        upload_time = self.simulate_upload_time(client)

        if client.id not in self.client_times:
            self.client_times[client.id] = []

        self.client_times[client.id].append((train_time, upload_time))
        
        self.agg_clients.append(client)  # add the client to the list of clients for aggregation

        with self.lock:
            self.count += 1
            self.aggregated_client_count += 1
            if self.aggregated_client_count == self.k and self.count != self.total_len:
                self.receive_models(req_clients=self.agg_clients)
                self.aggregate_parameters()
                self.aggregated_client_count = 0
                self.has_aggregated_first_k_clients = True

    def classify_clients_into_batches(self, k, selected_clients):
        """Classify clients into batches based on their average training and uploading times."""
        if k != 0 and len(self.client_times) != 0:
            avg_times = {client_id: np.mean(times, axis=0) for client_id, times in self.client_times.items()}
            sorted_clients = sorted(avg_times.items(), key=lambda item: item[1][0] + item[1][1])
            batch_size = len(sorted_clients) // 2
            fast_batch_ids = [client_id for client_id, _ in sorted_clients[:batch_size]]
            slow_batch_ids = [client_id for client_id, _ in sorted_clients[batch_size:]]

            self.fast_batch = [client for client in selected_clients if client.id in fast_batch_ids]
            self.slow_batch = [client for client in selected_clients if client.id in slow_batch_ids]

            for client in self.fast_batch:
                print("fast_batch client_id:", client.id)
            for client in self.slow_batch:
                print("slow_batch client_id:", client.id)

            return self.slow_batch, self.fast_batch

        else:
            batch_size = len(selected_clients) // 2
            self.fast_batch = [client for client in selected_clients[:batch_size] if isinstance(client, clientAVG)]
            self.slow_batch = [client for client in selected_clients[batch_size:] if isinstance(client, clientAVG)]

            for client in self.fast_batch:
                print("fast_batch client_id:", client.id)
            for client in self.slow_batch:
                print("slow_batch client_id:", client.id)

            print(f"Classified {len(self.fast_batch)} clients as fast and {len(self.slow_batch)} clients as slow.")
            return self.slow_batch, self.fast_batch

    def simulate_upload_time(self, client):
        # Placeholder method to simulate upload time
        return random.uniform(0.1, 1.0)
