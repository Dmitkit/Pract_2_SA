import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import math
import time
import random
import json
import os
from tkinter import filedialog

class GraphApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Алгоритм ближайшего соседа для задачи коммивояжёра")

        root.geometry("1920x1080")

        self.control_panel = tk.Frame(self.root, width=200)
        self.control_panel.grid(row=0, column=0, padx=5, pady=5, sticky="n")

        self.calc_button = tk.Button(self.control_panel, text="Рассчитать путь", command=self.calculate_tsp)
        self.calc_button.grid(row=0, column=0, padx=5, pady=5)

        self.default_button = tk.Button(self.control_panel, text="Пример по умолчанию", command=self.load_default_example)
        self.default_button.grid(row=1, column=0, padx=5, pady=5)

        self.load_file_button = tk.Button(self.control_panel, text="Загрузить граф\nиз файла", command=self.load_graph_from_file)
        self.load_file_button.grid(row=1, column=1, padx=5, pady=5)

        self.path_label = tk.Label(self.control_panel, text="Путь:")
        self.path_label.grid(row=2, column=0, padx=5, pady=5, sticky="e")

        self.path_display = tk.Entry(self.control_panel, width=25)
        self.path_display.grid(row=2, column=1, padx=5, pady=5)

        self.length_label = tk.Label(self.control_panel, text="Длина пути:")
        self.length_label.grid(row=3, column=0, padx=5, pady=5, sticky="e")

        self.length_display = tk.Entry(self.control_panel, width=25)
        self.length_display.grid(row=3, column=1, padx=5, pady=5)

        self.clear_button = tk.Button(self.control_panel, text="Очистить", command=self.clear_all)
        self.clear_button.grid(row=4, column=0, padx=5, pady=5)

        self.calc_button_mod = tk.Button(self.control_panel, text="Рассчитать путь\n(модифиц.)", command=self.calculate_tsp_improved)
        self.calc_button_mod.grid(row=4, column=1, padx=5, pady=5)

        self.time_label = tk.Label(self.control_panel, text="Затраченное\nвремя (сек):")
        self.time_label.grid(row=5, column=0, padx=5, pady=5)

        self.time_display = tk.Entry(self.control_panel, width=25)
        self.time_display.grid(row=5, column=1, padx=5, pady=5)

        self.fsa_button = tk.Button(self.control_panel, text="Оптимизировать FSA", command=self.run_fsa_optimization)
        self.fsa_button.grid(row=6, column=0, padx=5, pady=5)

        self.sa_button = tk.Button(self.control_panel, text="Оптимизировать SA", command=self.run_sa_optimization)
        self.sa_button.grid(row=6, column=1, padx=5, pady=5)

        self.sa_params_frame = tk.LabelFrame(self.control_panel, text="Параметры SA", padx=5, pady=5)
        self.sa_params_frame.grid(row=7, column=0, columnspan=2, padx=5, pady=5, sticky="we")

        tk.Label(self.sa_params_frame, text="Нач. темп.:").grid(row=0, column=0, sticky="w")
        self.sa_temp_entry = tk.Entry(self.sa_params_frame, width=8)
        self.sa_temp_entry.grid(row=0, column=1, padx=5)
        self.sa_temp_entry.insert(0, "100")

        tk.Label(self.sa_params_frame, text="Коэф. охл.:").grid(row=1, column=0, sticky="w")
        self.sa_cooling_entry = tk.Entry(self.sa_params_frame, width=8)
        self.sa_cooling_entry.grid(row=1, column=1, padx=5)
        self.sa_cooling_entry.insert(0, "0.995")

        tk.Label(self.sa_params_frame, text="Итераций:").grid(row=2, column=0, sticky="w")
        self.sa_iter_entry = tk.Entry(self.sa_params_frame, width=8)
        self.sa_iter_entry.grid(row=2, column=1, padx=5)
        self.sa_iter_entry.insert(0, "5000")

        self.fsa_params_frame = tk.LabelFrame(self.control_panel, text="Параметры FSA", padx=5, pady=5)
        self.fsa_params_frame.grid(row=8, column=0, columnspan=2, padx=5, pady=5, sticky="we")

        tk.Label(self.fsa_params_frame, text="Нач. темп.:").grid(row=0, column=0, sticky="w")
        self.fsa_temp_entry = tk.Entry(self.fsa_params_frame, width=8)
        self.fsa_temp_entry.grid(row=0, column=1, padx=5)
        self.fsa_temp_entry.insert(0, "100")

        tk.Label(self.fsa_params_frame, text="Итераций:").grid(row=1, column=0, sticky="w")
        self.fsa_iter_entry = tk.Entry(self.fsa_params_frame, width=8)
        self.fsa_iter_entry.grid(row=1, column=1, padx=5)
        self.fsa_iter_entry.insert(0, "5000")

        self.aco_params_frame = tk.LabelFrame(self.control_panel, text="Параметры ACO", padx=5, pady=5)
        self.aco_params_frame.grid(row=9, column=0, columnspan=2, padx=5, pady=5, sticky="we")

        tk.Label(self.aco_params_frame, text="Альфа (феромон):").grid(row=0, column=0, sticky="w")
        self.aco_alpha_entry = tk.Entry(self.aco_params_frame, width=8)
        self.aco_alpha_entry.grid(row=0, column=1, padx=5)
        self.aco_alpha_entry.insert(0, "1.0")

        tk.Label(self.aco_params_frame, text="Бета (длина):").grid(row=1, column=0, sticky="w")
        self.aco_beta_entry = tk.Entry(self.aco_params_frame, width=8)
        self.aco_beta_entry.grid(row=1, column=1, padx=5)
        self.aco_beta_entry.insert(0, "5.0")

        tk.Label(self.aco_params_frame, text="Феромон (Q):").grid(row=2, column=0, sticky="w")
        self.aco_q_entry = tk.Entry(self.aco_params_frame, width=8)
        self.aco_q_entry.grid(row=2, column=1, padx=5)
        self.aco_q_entry.insert(0, "100")

        tk.Label(self.aco_params_frame, text="Испарение (rho):").grid(row=3, column=0, sticky="w")
        self.aco_evap_entry = tk.Entry(self.aco_params_frame, width=8)
        self.aco_evap_entry.grid(row=3, column=1, padx=5)
        self.aco_evap_entry.insert(0, "0.5")

        tk.Label(self.aco_params_frame, text="Муравьёв:").grid(row=4, column=0, sticky="w")
        self.aco_ant_entry = tk.Entry(self.aco_params_frame, width=8)
        self.aco_ant_entry.grid(row=4, column=1, padx=5)
        self.aco_ant_entry.insert(0, "20")

        tk.Label(self.aco_params_frame, text="Итераций:").grid(row=5, column=0, sticky="w")
        self.aco_iter_entry = tk.Entry(self.aco_params_frame, width=8)
        self.aco_iter_entry.grid(row=5, column=1, padx=5)
        self.aco_iter_entry.insert(0, "100")

        self.aco_button = tk.Button(self.control_panel, text="Рассчитать путь\n(ACO)", command=self.run_aco)
        self.aco_button.grid(row=10, column=0, columnspan=2, padx=5, pady=5)

        ttk.Separator(self.root, orient=tk.VERTICAL).grid(row=0, column=1, rowspan=2, sticky="ns", padx=5)

        self.canvas = tk.Canvas(self.root, width=500, height=350, bg='white')
        self.canvas.grid(row=0, column=2, padx=5)
        tk.Label(self.root, text="Исходный граф").grid(row=0, column=2, sticky="n")

        self.result_canvas = tk.Canvas(self.root, width=500, height=350, bg='white')
        self.result_canvas.grid(row=1, column=2, padx=5)
        tk.Label(self.root, text="Гамильтонов цикл").grid(row=1, column=2, sticky="n")

        ttk.Separator(self.root, orient=tk.VERTICAL).grid(row=0, column=3, rowspan=2, sticky="ns", padx=5)

        self.table_frame = tk.Frame(self.root, width=200, height=300)
        self.table_frame.grid(row=0, column=4, rowspan=2, padx=5, pady=5, sticky="n")

        self.edge_table = ttk.Treeview(self.table_frame, columns=("From", "To", "Weight"), show="headings")
        self.edge_table.heading("From", text="От", anchor="w")
        self.edge_table.heading("To", text="К", anchor="w")
        self.edge_table.heading("Weight", text="Вес", anchor="w")
        self.edge_table.column("From", width=50)
        self.edge_table.column("To", width=50)
        self.edge_table.column("Weight", width=60)
        self.edge_table.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")

        self.scrollbar = ttk.Scrollbar(self.table_frame, orient="vertical", command=self.edge_table.yview)
        self.scrollbar.grid(row=0, column=1, sticky="ns")
        self.edge_table.configure(yscrollcommand=self.scrollbar.set)

        self.edge_table.bind("<Double-1>", self.on_table_double_click)

        self.node_radius = 15
        self.nodes = []
        self.edges = []
        self.selected_node = None
        self.node_positions = {}

        self.canvas.bind("<Button-1>", self.on_canvas_click)

        self.current_path = []
        self.current_length = 0

    def run_fsa_optimization(self):
        """ Быстрая симуляция отжига (Fast Simulated Annealing) """
        if not self.current_path or len(self.current_path) < 3:
            messagebox.showinfo("Информация", "Сначала постройте гамильтонов цикл (минимум 3 узла).")
            return
        edge_dict = {(e['from'], e['to']): e['weight'] for e in self.edges}

        try:
            T0 = float(self.fsa_temp_entry.get())
            max_iterations = int(self.fsa_iter_entry.get())
        except ValueError:
            messagebox.showerror("Ошибка", "Некорректные параметры FSA")
            return

        start_time = time.time()
        optimized_path, optimized_length = self.fast_simulated_annealing(
            self.current_path.copy(),
            edge_dict,
            T0,
            max_iterations
        )
        elapsed_time = time.time() - start_time

        if optimized_path:
            self.current_path = optimized_path
            self.current_length = optimized_length
            self.display_path(optimized_path, optimized_length)
            self.draw_result_path(optimized_path)
            self.time_display.delete(0, tk.END)
            self.time_display.insert(0, f"{elapsed_time:.4f}")

    def fast_simulated_annealing(self, path, edge_weights, T0, max_iterations):
        k_temp = 0.5
        eps = 1e-08

        current_path = path[:-1] if path[0] == path[-1] else path.copy()

        best_path = current_path[:]
        best_cost = self.path_cost(best_path, edge_weights)

        current_cost = best_cost
        no_improvement_count = 0
        stagnation_limit = 300  

        for k in range(1, max_iterations + 1):
            T = 0

            if k < 3:
                T = T0
            else:
                T = T0 / (k + 1)

            T = max(T, eps)

            neighbor = self.generate_neighbor_swap(current_path, edge_weights)
            if neighbor is None:
                neighbor = self.random_kick(current_path)

            neighbor_cost = self.path_cost(neighbor, edge_weights)
            delta = neighbor_cost - current_cost

            if delta < 0:

                current_path = neighbor
                current_cost = neighbor_cost
                no_improvement_count = 0
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_path = current_path[:]
            else:
                if random.random() < math.exp(-delta / (k_temp * T)):
                    current_path = neighbor
                    current_cost = neighbor_cost
                no_improvement_count += 1
                if no_improvement_count > stagnation_limit:
                    current_path = self.random_kick(current_path)
                    current_cost = self.path_cost(current_path, edge_weights)
                    no_improvement_count = 0

        return best_path + [best_path[0]], best_cost

    def run_sa_optimization(self):
        """ Классический SA с геометрическим охлаждением """
        if not self.current_path or len(self.current_path) < 3:
            messagebox.showinfo("Информация", "Сначала постройте гамильтонов цикл (минимум 3 узла).")
            return
        edge_dict = {(e['from'], e['to']): e['weight'] for e in self.edges}

        try:
            T0 = float(self.sa_temp_entry.get())
            cooling_rate = float(self.sa_cooling_entry.get())
            max_iterations = int(self.sa_iter_entry.get())
        except ValueError:
            messagebox.showerror("Ошибка", "Некорректные параметры SA")
            return

        start_time = time.time()
        optimized_path, optimized_length = self.simulated_annealing(
            self.current_path.copy(),
            edge_dict,
            T0,
            cooling_rate,
            max_iterations
        )
        elapsed_time = time.time() - start_time

        if optimized_path:
            self.current_path = optimized_path
            self.current_length = optimized_length
            self.display_path(optimized_path, optimized_length)
            self.draw_result_path(optimized_path)
            self.time_display.delete(0, tk.END)
            self.time_display.insert(0, f"{elapsed_time:.4f}")

    def simulated_annealing(self, path, edge_weights, T0, cooling_rate, max_iterations):
        """
        Классическая SA:
        - Температуру понижаем геометрически: T = T * cooling_rate.
        - Генерация соседей (2-opt). При большом застое — случайный «kick».
        """
        current_path = path[:-1] if path[0] == path[-1] else path.copy()
        best_path = current_path[:]
        best_cost = self.path_cost(best_path, edge_weights)

        current_cost = best_cost
        T = T0
        no_improvement_count = 0
        stagnation_limit = 300

        for _ in range(max_iterations):
            neighbor = self.generate_neighbor_swap(current_path, edge_weights)
            if neighbor is None:
                neighbor = self.random_kick(current_path)

            neighbor_cost = self.path_cost(neighbor, edge_weights)
            delta = neighbor_cost - current_cost

            if delta < 0:
                current_path = neighbor
                current_cost = neighbor_cost
                no_improvement_count = 0
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_path = current_path[:]
            else:
                if random.random() < math.exp(-delta / max(T, 1e-12)):
                    current_path = neighbor
                    current_cost = neighbor_cost
                no_improvement_count += 1
                if no_improvement_count > stagnation_limit:
                    current_path = self.random_kick(current_path)
                    current_cost = self.path_cost(current_path, edge_weights)
                    no_improvement_count = 0

            T *= cooling_rate
            if T < 1e-9:  
                break

        return best_path + [best_path[0]], best_cost

    def generate_neighbor_swap(self, path, edge_weights, max_attempts=20):
        if len(path) < 3:
            return None

        n = len(path)
        for _ in range(max_attempts):
            i = random.randint(1, n - 1)
            j = random.randint(1, n - 1)
            if i == j:
                continue

            new_path = path.copy()
            new_path[i], new_path[j] = new_path[j], new_path[i]

            valid = self.is_valid_path(new_path, edge_weights)
            valid = valid and ((new_path[-1], new_path[0]) in edge_weights)

            if valid:
                return new_path

        return None

    def is_valid_path(self, path, edge_weights):
        for i in range(len(path)-1):
            if (path[i], path[i+1]) not in edge_weights:
                return False
        return True

    def random_kick(self, path):
        """
        Небольшая рандомная перестановка (shake),
        чтобы «выпрыгнуть» из застоя.
        """
        new_path = path[:]

        length = len(new_path)
        if length < 4:
            return new_path

        swap_indices = random.sample(range(1, length), min(4, length - 1))
        sub = [new_path[idx] for idx in swap_indices]
        random.shuffle(sub)
        for i, idx in enumerate(swap_indices):
            new_path[idx] = sub[i]
        return new_path

    def path_cost(self, path, edge_weights):
        total = 0.0
        for i in range(len(path)-1):
            w = edge_weights.get((path[i], path[i+1]), float('inf'))
            total += w
        return total

    def clear_all(self):
        self.time_display.delete(0, tk.END)
        self.canvas.delete("all")
        self.result_canvas.delete("all")
        self.path_display.delete(0, tk.END)
        self.length_display.delete(0, tk.END)
        for item in self.edge_table.get_children():
            self.edge_table.delete(item)
        self.nodes.clear()
        self.edges.clear()
        self.node_positions.clear()

    def on_canvas_click(self, event):
        x, y = event.x, event.y
        node = self.get_node_at_position(x, y)
        if node:
            if self.selected_node:
                self.create_edge(self.selected_node, node)
                self.selected_node = None
            else:
                self.selected_node = node
        else:
            if self.selected_node:
                self.selected_node = None
            self.create_node(x, y)

    def create_node(self, x, y):
        node_id = len(self.nodes) + 1
        node = {'id': node_id, 'x': x, 'y': y}
        self.nodes.append(node)
        self.node_positions[node_id] = (x, y)
        self.canvas.create_oval(
            x - self.node_radius, y - self.node_radius,
            x + self.node_radius, y + self.node_radius,
            fill="blue", tags=f"node{node_id}"
        )
        self.canvas.create_text(x, y, text=str(node_id), fill="white")

    def create_edge(self, from_node, to_node):
        from_x, from_y = from_node['x'], from_node['y']
        to_x, to_y = to_node['x'], to_node['y']
        edge_id = len(self.edges) + 1
        self.canvas.create_line(from_x, from_y, to_x, to_y, arrow=tk.LAST, tags=f"edge{edge_id}")
        weight = self.calculate_distance(from_node, to_node)
        self.edges.append({'id': edge_id, 'from': from_node['id'], 'to': to_node['id'], 'weight': weight})
        self.edge_table.insert("", "end", values=(from_node['id'], to_node['id'], weight))

    def get_node_at_position(self, x, y):
        for node in self.nodes:
            node_x, node_y = node['x'], node['y']
            if ((node_x - self.node_radius) < x < (node_x + self.node_radius)
                    and (node_y - self.node_radius) < y < (node_y + self.node_radius)):
                return node
        return None

    def calculate_distance(self, from_node, to_node):
        x1, y1 = from_node['x'], from_node['y']
        x2, y2 = to_node['x'], to_node['y']
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    def on_table_double_click(self, event):
        region = self.edge_table.identify_region(event.x, event.y)
        if region == "cell":
            column = self.edge_table.identify_column(event.x)
            if column == "#3":
                item = self.edge_table.selection()[0]
                column_box = self.edge_table.bbox(item, column)
                current_value = self.edge_table.item(item, "values")[2]
                entry = ttk.Entry(self.edge_table, width=8)
                entry.place(x=column_box[0], y=column_box[1],
                            width=column_box[2], height=column_box[3])
                entry.insert(0, current_value)
                entry.focus_set()
                entry.bind("<FocusOut>", lambda e: self.update_edge_weight(item, entry))
                entry.bind("<Return>", lambda e: self.update_edge_weight(item, entry))

    def update_edge_weight(self, item, entry):
        if not self.edge_table.exists(item):
            entry.destroy()
            return
        new_weight = entry.get()
        try:
            new_weight = float(new_weight)
            if new_weight < 0:
                raise ValueError("Вес не может быть отрицательным")
        except ValueError:
            tk.messagebox.showerror("Ошибка", "Введите корректное число для веса")
            return
        values = list(self.edge_table.item(item, "values"))
        values[2] = new_weight
        self.edge_table.item(item, values=values)
        from_node_id = int(values[0])
        to_node_id = int(values[1])
        for edge in self.edges:
            if edge['from'] == from_node_id and edge['to'] == to_node_id:
                edge['weight'] = new_weight
                break
        entry.destroy()

    def load_default_example(self):
        file_path = os.path.join("graphs", "graph_default.json")
        self.load_graph(file_path)

    def load_graph(self, file_path):
        self.nodes.clear()
        self.edges.clear()
        self.node_positions.clear()
        self.canvas.delete("all")
        self.result_canvas.delete("all")
        self.edge_table.delete(*self.edge_table.get_children())
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for node in data["default_nodes"]:
            self.create_node(node["x"], node["y"])
            self.canvas.create_text(node["x"], node["y"] - 25, fill="black")
        for edge in data["default_edges"]:
            from_node = next(n for n in self.nodes if n["id"] == edge["from"])
            to_node = next(n for n in self.nodes if n["id"] == edge["to"])
            self.create_edge(from_node, to_node)
        for edge in self.edges:
            for e in data["default_edges"]:
                if edge["from"] == e["from"] and edge["to"] == e["to"]:
                    edge["weight"] = e["weight"]
                    for item in self.edge_table.get_children():
                        values = self.edge_table.item(item, "values")
                        if int(values[0]) == edge["from"] and int(values[1]) == edge["to"]:
                            self.edge_table.item(item, values=(values[0], values[1], e["weight"]))
                    break

    def load_graph_from_file(self):
        file_path = filedialog.askopenfilename(title="Выберите файл графа",
                                               filetypes=[("JSON files", "*.json")])
        if file_path:
            self.load_graph(file_path)

    def calculate_tsp(self):
        start_time = time.time()
        if len(self.nodes) < 2:
            self.time_display.delete(0, tk.END)
            self.time_display.insert(0, "0.0000")
            return

        edges_from_node = {}
        for edge in self.edges:
            from_node = edge['from']
            if from_node not in edges_from_node:
                edges_from_node[from_node] = []
            edges_from_node[from_node].append(edge)

        start_node = self.nodes[0]
        current_node_id = start_node['id']
        path = [current_node_id]
        total_length = 0
        visited = {node['id']: False for node in self.nodes}
        visited[current_node_id] = True

        for _ in range(len(self.nodes) - 1):
            available_edges = edges_from_node.get(current_node_id, [])
            valid_edges = [edge for edge in available_edges if not visited[edge['to']]]
            if not valid_edges:
                break
            min_edge = min(valid_edges, key=lambda x: x['weight'])
            next_node_id = min_edge['to']
            path.append(next_node_id)
            total_length += min_edge['weight']
            visited[next_node_id] = True
            current_node_id = next_node_id

        return_edges = [edge for edge in edges_from_node.get(current_node_id, [])
                        if edge['to'] == start_node['id']]
        if return_edges:
            total_length += return_edges[0]['weight']
            path.append(start_node['id'])

        end_time = time.time()
        elapsed_time = end_time - start_time
        self.time_display.delete(0, tk.END)
        self.time_display.insert(0, f"{elapsed_time:.4f}")

        self.current_path = path
        self.current_length = total_length
        self.display_path(path, total_length)
        self.draw_result_path(path)

    def calculate_tsp_improved(self):
        def run_nearest_neighbor(start_node):
            edges_from_node = {}
            for edge in self.edges:
                if edge['from'] not in edges_from_node:
                    edges_from_node[edge['from']] = []
                edges_from_node[edge['from']].append(edge)

            current_node_id = start_node['id']
            path = [current_node_id]
            total_length = 0
            visited = {node['id']: False for node in self.nodes}
            visited[current_node_id] = True

            for _ in range(len(self.nodes) - 1):
                available_edges = edges_from_node.get(current_node_id, [])
                valid_edges = [edge for edge in available_edges if not visited[edge['to']]]
                if not valid_edges:
                    break
                min_edge = min(valid_edges, key=lambda x: x['weight'])
                next_node_id = min_edge['to']
                path.append(next_node_id)
                total_length += min_edge['weight']
                visited[next_node_id] = True
                current_node_id = next_node_id

            return_edges = [edge for edge in edges_from_node.get(current_node_id, [])
                            if edge['to'] == start_node['id']]
            if return_edges:
                total_length += return_edges[0]['weight']
                path.append(start_node['id'])

            return path, total_length

        start_time = time.time()
        if len(self.nodes) < 2:
            self.time_display.delete(0, tk.END)
            self.time_display.insert(0, "0.0000")
            return

        best_path = None
        best_length = float('inf')
        for start_node in self.nodes:
            path, length = run_nearest_neighbor(start_node)
            if length < best_length:
                best_path = path
                best_length = length

        end_time = time.time()
        elapsed_time = end_time - start_time
        self.time_display.delete(0, tk.END)
        self.time_display.insert(0, f"{elapsed_time:.4f}")

        self.current_path = best_path
        self.current_length = best_length
        self.display_path(best_path, best_length)
        self.draw_result_path(best_path)

    def display_path(self, path, length):
        self.path_display.delete(0, tk.END)
        self.path_display.insert(0, " -> ".join(map(str, path)))
        self.length_display.delete(0, tk.END)
        self.length_display.insert(0, str(length))

    def draw_result_path(self, path):
        self.result_canvas.delete("all")
        edge_ids = {(e['from'], e['to']): e for e in self.edges}
        for i in range(len(path)-1):
            from_id, to_id = path[i], path[i+1]
            if (from_id, to_id) in edge_ids:
                from_node = next(n for n in self.nodes if n['id'] == from_id)
                to_node = next(n for n in self.nodes if n['id'] == to_id)
                x1, y1 = from_node['x'], from_node['y']
                x2, y2 = to_node['x'], to_node['y']
                self.result_canvas.create_line(x1, y1, x2, y2, arrow=tk.LAST, fill="green", width=2)
        for node in self.nodes:
            x, y = node['x'], node['y']
            self.result_canvas.create_oval(x - self.node_radius, y - self.node_radius,
                                           x + self.node_radius, y + self.node_radius,
                                           fill="blue")
            self.result_canvas.create_text(x, y, text=str(node['id']), fill="white")

    def run_aco(self):
        """Запуск муравьиного алгоритма (ACO) для поиска гамильтонова цикла."""
        if not self.edges or len(self.edges) < 3:
            messagebox.showinfo("Информация", "Граф должен содержать хотя бы 3 ребра.")
            return

        try:
            alpha      = float(self.aco_alpha_entry.get())
            beta       = float(self.aco_beta_entry.get())
            q          = float(self.aco_q_entry.get())
            evaporation= float(self.aco_evap_entry.get())
            ant_count  = int(self.aco_ant_entry.get())
            iterations = int(self.aco_iter_entry.get())
        except ValueError:
            messagebox.showerror("Ошибка", "Некорректные параметры ACO.")
            return

        # 1) Собираем веса ребер
        edge_weights = {(e['from'], e['to']): e['weight'] for e in self.edges}

        # 2) Делаем список всех узлов из ключей edge_weights
        nodes = list({u for u, v in edge_weights.keys()} | {v for u, v in edge_weights.keys()})

        pheromones = {edge: 1.0 for edge in edge_weights}
        best_path = None
        best_cost = float('inf')

        start_time = time.time()
        for _ in range(iterations):
            all_paths = []
            for _ in range(ant_count):
                path = self.ant_colony_optimization(nodes, pheromones, edge_weights, alpha, beta)
                if path:
                    cost = self.path_cost(path, edge_weights)
                    all_paths.append((path, cost))
                    if cost < best_cost:
                        best_cost = cost
                        best_path = path

            # испарение
            for edge in pheromones:
                pheromones[edge] *= (1 - evaporation)
            # подкрепление
            for path, cost in all_paths:
                for i in range(len(path) - 1):
                    e = (path[i], path[i + 1])
                    if e in pheromones:
                        pheromones[e] += q / cost

        elapsed = time.time() - start_time
        if best_path:
            self.current_path   = best_path
            self.current_length = best_cost
            self.display_path(best_path, best_cost)
            self.draw_result_path(best_path)
            self.time_display.delete(0, tk.END)
            self.time_display.insert(0, f"{elapsed:.4f}")
        else:
            messagebox.showinfo("Информация", "Не удалось найти допустимый гамильтонов цикл.")

    def ant_colony_optimization(self, nodes, pheromones, edge_weights, alpha, beta):
        """
        Одна итерация ACO: конструируем маршрут муравья.
        Возвращает путь (с замыканием на старт) или None, если цикл не построился.
        """
        if not nodes:
            return None

        start = random.choice(nodes)
        path = [start]
        unvisited = set(nodes) - {start}
        current = start

        while unvisited:
            probabilities = []
            total = 0.0

            for neighbor in list(unvisited):
                edge = (current, neighbor)
                if edge not in edge_weights:
                    continue
                pher = pheromones.get(edge, 1.0)
                dist = edge_weights[edge] or 1e-6
                score = (pher ** alpha) * ((1.0 / dist) ** beta)
                probabilities.append((neighbor, score))
                total += score

            if total == 0.0:
                return None

            r = random.uniform(0, total)
            cum = 0.0
            for neighbor, score in probabilities:
                cum += score
                if r <= cum:
                    path.append(neighbor)
                    unvisited.remove(neighbor)
                    current = neighbor
                    break

        # замыкаем цикл, если есть ребро обратно
        if (path[-1], path[0]) in edge_weights:
            path.append(path[0])
            return path

        return None


    def select_next_node(self, current, unvisited, pheromones, edge_weights, alpha, beta):
        probabilities = {}
        total_prob = 0.0

        for next_node in unvisited:
            edge = (current, next_node) if (current, next_node) in edge_weights else (next_node, current)
            pheromone = pheromones.get(edge, 1.0)
            distance = edge_weights.get(edge, float('inf'))
            prob = (pheromone ** alpha) * ((1.0 / distance) ** beta)
            probabilities[next_node] = prob
            total_prob += prob

        if total_prob == 0:
            return random.choice(list(unvisited))

        # Нормируем вероятности
        for next_node in probabilities:
            probabilities[next_node] /= total_prob

        # Выбираем следующий узел на основе вероятности
        return random.choices(list(probabilities.keys()), weights=probabilities.values(), k=1)[0]

    def construct_graph(self):
        """
        Считывает из таблицы self.edge_table ребра и заполняет self.nodes и self.edges.
        """
        self.nodes = []
        self.edges = []

        for iid in self.edge_table.get_children():
            frm, to, w = self.edge_table.item(iid)['values']
            weight = float(w)
            self.nodes.append(frm)
            self.nodes.append(to)
            self.edges.append({'from': frm, 'to': to, 'weight': weight})

        # уникальность узлов
        self.nodes = list(set(self.nodes))


if __name__ == "__main__":
    root = tk.Tk()
    app = GraphApp(root)
    root.mainloop()