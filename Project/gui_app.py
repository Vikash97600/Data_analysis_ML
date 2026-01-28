import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import os
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

class MatrixGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Data Analysis & Visualization Tool")
        self.root.geometry("1600x900") 
        self.root.configure(bg="#e8ecf1")
        
        
        self.bg_color = "#e8ecf1"
        self.card_bg = "#f0f4f9"
        self.primary_color = "#2d5f8d"
        self.secondary_color = "#4a7ba7"
        self.accent_color = "#ff6b6b"
        self.success_color = "#51cf66"
        self.warning_color = "#f59f00"
        self.text_color = "#2c3e50"
        self.text_light = "#5a6c7d"
        self.shadow_dark = "#c8d0da"
        self.shadow_light = "#ffffff"
        
       
        self.current_theme = "corporate"
        self.current_font_name = "Segoe UI"
        self.available_fonts = ["Segoe UI", "Consolas", "Arial", "Times New Roman", "Courier New", "Georgia", "Calibri", "Verdana"]
        
        
        self.setup_styles()
        
        self.matrix_a = None
        self.matrix_b = None
        self.df = None
        self.panel_open = True
        
        self.create_ui()
    
    def setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        
        
        style.configure('TNotebook', background=self.bg_color, borderwidth=0)
        style.configure('TNotebook.Tab', 
                       padding=[25, 8], 
                       background=self.card_bg, 
                       foreground=self.text_color,
                       font=('Segoe UI', 10, 'bold'), 
                       relief='flat', 
                       borderwidth=2,
                       bordercolor=self.shadow_dark)
        style.map('TNotebook.Tab', 
                 background=[('selected', self.primary_color)], 
                 foreground=[('selected', '#ffffff')],
                 relief=[('selected', 'flat')])
        
        
        style.configure('TFrame', background=self.bg_color)
        style.configure('Card.TFrame', background=self.card_bg, relief='flat', borderwidth=0)
        
        
        style.configure('TLabelFrame', 
                       background=self.bg_color, 
                       foreground=self.primary_color,
                       borderwidth=1, 
                       relief='solid',
                       padding=15,
                       bordercolor=self.shadow_dark)
        style.configure('TLabelFrame.Label', 
                       background=self.bg_color, 
                       foreground=self.primary_color,
                       font=('Segoe UI', 11, 'bold'))
        
      
        style.configure('TLabel', background=self.bg_color, foreground=self.text_color, font=('Segoe UI', 9))
        style.configure('Header.TLabel', background=self.bg_color, foreground=self.primary_color, font=('Segoe UI', 14, 'bold'))
        style.configure('Success.TLabel', background=self.bg_color, foreground=self.success_color, font=('Segoe UI', 9, 'bold'))
        
        
        style.configure('TButton', 
                       background=self.primary_color,
                       foreground='#ffffff',
                       borderwidth=0,
                       focuscolor='none',
                       padding=12,
                       font=('Segoe UI', 9, 'bold'),
                       relief='flat')
        style.map('TButton',
                 background=[('active', self.secondary_color), ('pressed', '#2c5282')],
                 foreground=[('active', '#ffffff'), ('pressed', '#ffffff')])
        
        
        style.configure('Secondary.TButton',
                       background=self.card_bg,
                       foreground=self.text_color,
                       borderwidth=1,
                       bordercolor=self.shadow_dark)
        style.map('Secondary.TButton',
                 background=[('active', self.shadow_light)],
                 foreground=[('active', self.primary_color)])
        
       
        style.configure('TEntry', 
                       fieldbackground='#ffffff',
                       foreground=self.text_color,
                       borderwidth=1,
                       relief='solid',
                       padding=8,
                       font=('Segoe UI', 9),
                       bordercolor=self.shadow_dark)
        
        
        style.configure('TScrollbar',
                       background=self.primary_color,
                       troughcolor=self.card_bg,
                       borderwidth=0,
                       arrowsize=12)
    
    def create_ui(self):
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=0, pady=0)
        
        
        header_frame = ttk.Frame(main_container, style='Card.TFrame')
        header_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=(10, 5))
        
        
        title_frame = ttk.Frame(header_frame)
        title_frame.pack(side=tk.LEFT, padx=15)
        
        title_label = ttk.Label(title_frame, 
                               text="📊 QSkill Data Analysis Project", 
                               font=('Segoe UI', 18, 'bold'),
                               foreground=self.primary_color)
        title_label.pack(side=tk.LEFT)
        
        version_badge = ttk.Label(title_frame,
                                 text=" ",
                                 font=('Segoe UI', 9),
                                 foreground=self.text_light)
        version_badge.pack(side=tk.LEFT, padx=(10, 0))
        
    
        button_frame = ttk.Frame(header_frame)
        button_frame.pack(side=tk.RIGHT, padx=15)
        
        quick_actions = [
            ("📊 Quick Plot", lambda: self.notebook.select(1)),
            ("🤖 Train Model", lambda: self.notebook.select(2)),
            ("🔄 Refresh", self.refresh_app)
        ]
        
        for text, command in quick_actions:
            btn = ttk.Button(button_frame, 
                            text=text, 
                            command=command,
                            style='Secondary.TButton')
            btn.pack(side=tk.LEFT, padx=3)
        
        ttk.Separator(button_frame, orient='vertical').pack(side=tk.LEFT, padx=10, fill=tk.Y)
        
        ttk.Button(button_frame, 
                  text="ℹ️ About", 
                  command=self.show_about,
                  style='Secondary.TButton').pack(side=tk.LEFT, padx=3)
        ttk.Button(button_frame, 
                  text="❌ Exit", 
                  command=self.root.quit,
                  style='Secondary.TButton').pack(side=tk.LEFT, padx=3)
        
        
        content_container = ttk.Frame(main_container)
        content_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        
       
        self.side_panel = ttk.LabelFrame(content_container, 
                                        text=" Navigation Panel ",
                                        padding=20,
                                        width=220)  
        self.side_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        self.side_panel.pack_propagate(False)  
        
        
        nav_header = ttk.Label(self.side_panel, 
                              text="🔍 Explore Features",
                              font=('Segoe UI', 12, 'bold'),
                              foreground=self.primary_color)
        nav_header.pack(anchor=tk.W, pady=(0, 15))
        
        nav_sections = [
            ("🧮 Matrix Operations", "Perform matrix calculations", 0),
            ("📊 Data Visualization", "Create charts and graphs", 1),
            ("🤖 ML Regression", "Train predictive models", 2)
        ]
        
        for label, tooltip, tab_idx in nav_sections:
            frame = ttk.Frame(self.side_panel)
            frame.pack(fill=tk.X, pady=4)
            
            btn = ttk.Button(frame,
                           text=label,
                           command=lambda idx=tab_idx: self.notebook.select(idx),
                           style='Secondary.TButton')
            btn.pack(fill=tk.X)
            
            tip = ttk.Label(frame,
                          text=tooltip,
                          font=('Segoe UI', 8),
                          foreground=self.text_light)
            tip.pack(anchor=tk.W, padx=5)
        
        ttk.Separator(self.side_panel, orient='horizontal').pack(fill=tk.X, pady=15)
        
        
        status_frame = ttk.LabelFrame(self.side_panel, 
                                     text=" System Status ",
                                     padding=10)
        status_frame.pack(fill=tk.X, pady=5)
        
        self.status_label = ttk.Label(status_frame,
                                     text="✓ All systems ready",
                                     foreground=self.success_color,
                                     font=('Segoe UI', 9, 'bold'))
        self.status_label.pack(anchor=tk.W)
        

       
        settings_header = ttk.Label(self.side_panel,
                                   text="⚙️ Quick Settings",
                                   font=('Segoe UI', 12, 'bold'),
                                   foreground=self.primary_color)
        settings_header.pack(anchor=tk.W, pady=(0, 10))
        
        

        theme_frame = ttk.Frame(self.side_panel)
        theme_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(theme_frame, text="Theme:", font=('Segoe UI', 9)).pack(side=tk.LEFT)
        
        theme_var = tk.StringVar(value="corporate")
        theme_combo = ttk.Combobox(theme_frame,
                                  textvariable=theme_var,
                                  values=["corporate", "light", "dark"],
                                  state="readonly",
                                  width=12)
        theme_combo.pack(side=tk.RIGHT)
        theme_combo.bind('<<ComboboxSelected>>', lambda e: self.apply_theme(theme_var.get()))
        
      
      

        font_frame = ttk.Frame(self.side_panel)
        font_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(font_frame, text="Font:", font=('Segoe UI', 9)).pack(side=tk.LEFT)
        
        font_var = tk.StringVar(value="Segoe UI")
        font_combo = ttk.Combobox(font_frame,
                                 textvariable=font_var,
                                 values=self.available_fonts,
                                 state="readonly",
                                 width=12)
        font_combo.pack(side=tk.RIGHT)
        font_combo.bind('<<ComboboxSelected>>', lambda e: self.apply_font(font_var.get()))
        
        ttk.Separator(self.side_panel, orient='horizontal').pack(fill=tk.X, pady=15)
        
        
        

        help_frame = ttk.Frame(self.side_panel)
        help_frame.pack(fill=tk.X)
        
        ttk.Button(help_frame,
                  text="📖 User Guide",
                  command=self.show_guide,
                  style='Secondary.TButton').pack(fill=tk.X, pady=2)
        
        ttk.Button(help_frame,
                  text="🔄 Reset Layout",
                  command=self.reset_layout,
                  style='Secondary.TButton').pack(fill=tk.X, pady=2)
        
        
        

        content_frame = ttk.Frame(content_container)
        content_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        
        
        self.notebook = ttk.Notebook(content_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        
        
        self.matrix_frame = ttk.Frame(self.notebook)
        self.viz_frame = ttk.Frame(self.notebook)
        self.lr_frame = ttk.Frame(self.notebook)
        
        self.notebook.add(self.matrix_frame, text="  🧮 Matrix Operations  ")
        self.notebook.add(self.viz_frame, text="  📊 Data Visualization  ")
        self.notebook.add(self.lr_frame, text="  🤖 Machine Learning  ")
        
        
        
        self.create_matrix_tab()
        self.create_visualization_tab()
        self.create_linear_regression_tab()
        
        
        
        self.notebook.bind('<<NotebookTabChanged>>', self.on_tab_change)
    
    def on_tab_change(self, event):
        """Handle tab change events"""
        tab_names = ["Matrix Operations", "Data Visualization", "Machine Learning"]
        current_tab = self.notebook.index(self.notebook.select())
        self.status_label.config(text=f"Active: {tab_names[current_tab]}")
    
    def reset_layout(self):
        """Reset UI layout"""
        self.status_label.config(text="Layout reset ✓", foreground=self.success_color)
        messagebox.showinfo("Layout Reset", "UI layout has been reset to default.")
    
    def refresh_app(self):
        """Refresh application state"""
        self.status_label.config(text="Refreshed ✓", foreground=self.success_color)
        messagebox.showinfo("Refreshed", "Application refreshed successfully.")
    
    def create_matrix_tab(self):
        """Matrix operations tab"""
        main_container = ttk.Frame(self.matrix_frame)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        
        input_section = ttk.LabelFrame(main_container, text=" Matrix Input ", padding=20)
        input_section.pack(fill=tk.X, pady=(0, 15))
        
        
        matrix_grid = ttk.Frame(input_section)
        matrix_grid.pack(fill=tk.X, pady=10)
        
        
        matrix_a_frame = ttk.LabelFrame(matrix_grid, text=" Matrix A ", padding=15)
        matrix_a_frame.grid(row=0, column=0, padx=(0, 20), sticky="nsew")
        
        a_status_frame = ttk.Frame(matrix_a_frame)
        a_status_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(a_status_frame, text="Status:", font=('Segoe UI', 9)).pack(side=tk.LEFT)
        
        self.label_a = ttk.Label(a_status_frame,
                                text="Not loaded",
                                foreground=self.warning_color,
                                font=('Segoe UI', 9, 'bold'))
        self.label_a.pack(side=tk.RIGHT)
        
        ttk.Button(matrix_a_frame, text="📥 Load Matrix A", command=self.input_matrix_a, style='TButton').pack(fill=tk.X, pady=5)
        ttk.Button(matrix_a_frame, text="🔄 Transpose A", command=self.transpose_a, style='Secondary.TButton').pack(fill=tk.X, pady=3)
        ttk.Button(matrix_a_frame, text="📐 Determinant A", command=self.determinant_a, style='Secondary.TButton').pack(fill=tk.X, pady=3)
        
        
        matrix_b_frame = ttk.LabelFrame(matrix_grid, text=" Matrix B ", padding=15)
        matrix_b_frame.grid(row=0, column=1, sticky="nsew")
        
        b_status_frame = ttk.Frame(matrix_b_frame)
        b_status_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(b_status_frame, text="Status:", font=('Segoe UI', 9)).pack(side=tk.LEFT)
        
        self.label_b = ttk.Label(b_status_frame,
                                text="Not loaded",
                                foreground=self.warning_color,
                                font=('Segoe UI', 9, 'bold'))
        self.label_b.pack(side=tk.RIGHT)
        
        ttk.Button(matrix_b_frame, text="📥 Load Matrix B", command=self.input_matrix_b, style='TButton').pack(fill=tk.X, pady=5)
        ttk.Button(matrix_b_frame, text="🔄 Transpose B", command=self.transpose_b, style='Secondary.TButton').pack(fill=tk.X, pady=3)
        ttk.Button(matrix_b_frame, text="📐 Determinant B", command=self.determinant_b, style='Secondary.TButton').pack(fill=tk.X, pady=3)
        
        
        ops_section = ttk.LabelFrame(main_container, text=" Matrix Operations ", padding=20)
        ops_section.pack(fill=tk.X, pady=(0, 15))
        
        ops_frame = ttk.Frame(ops_section)
        ops_frame.pack(fill=tk.X)
        
        operations = [
            ("➕ Add", self.matrix_add, "A + B"),
            ("➖ Subtract", self.matrix_subtract, "A - B"),
            ("✖️ Multiply", self.matrix_multiply, "A × B"),
            ("🔄 Swap", self.swap_matrices, "Swap A ↔ B"),
            ("🧹 Clear", self.clear_matrices, "Clear All")
        ]
        
        for i, (text, command, tooltip) in enumerate(operations):
            frame = ttk.Frame(ops_frame)
            frame.grid(row=0, column=i, padx=5, sticky="nsew")
            
            btn = ttk.Button(frame, text=text, command=command, style='TButton')
            btn.pack(fill=tk.BOTH, expand=True)
            
            ttk.Label(frame, text=tooltip, font=('Segoe UI', 8), foreground=self.text_light).pack()
        
        
        results_section = ttk.LabelFrame(main_container, text=" Results ", padding=20)
        results_section.pack(fill=tk.BOTH, expand=True)
        
        
        result_header = ttk.Frame(results_section)
        result_header.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(result_header, text="Output:", font=('Segoe UI', 11, 'bold'), foreground=self.primary_color).pack(side=tk.LEFT)
        ttk.Button(result_header, text="Clear Output", command=lambda: self.display_result(""), style='Secondary.TButton').pack(side=tk.RIGHT)
        
        
        text_frame = ttk.Frame(results_section)
        text_frame.pack(fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(text_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.result_text = tk.Text(text_frame,
                                   height=15,
                                   yscrollcommand=scrollbar.set,
                                   font=("Consolas", 10),
                                   bg='#ffffff',
                                   fg=self.text_color,
                                   insertbackground=self.primary_color,
                                   borderwidth=1,
                                   relief='solid',
                                   padx=15,
                                   pady=15,
                                   highlightthickness=0,
                                   state=tk.DISABLED)
        self.result_text.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.result_text.yview)
    
    def create_visualization_tab(self):
        """Enhanced visualization tab with larger canvas area"""
        main_container = ttk.Frame(self.viz_frame)
        main_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        
        paned_window = ttk.PanedWindow(main_container, orient=tk.HORIZONTAL)
        paned_window.pack(fill=tk.BOTH, expand=True)
        
        
        left_panel = ttk.Frame(paned_window, width=300)
        left_panel.pack_propagate(False)
        
        
        load_section = ttk.LabelFrame(left_panel, text=" Data Source ", padding=15)
        load_section.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(load_section, text="📊 Sample Dataset", command=self.load_sample_data, style='TButton').pack(fill=tk.X, pady=5)
        ttk.Button(load_section, text="📁 Load CSV File", command=self.load_csv, style='TButton').pack(fill=tk.X, pady=5)
        
        self.data_label = ttk.Label(load_section, text="No data loaded", foreground=self.warning_color, font=('Segoe UI', 9, 'bold'))
        self.data_label.pack(pady=10)
        
        ttk.Button(load_section, text="👁️ Preview Data", command=self.preview_data, style='Secondary.TButton').pack(fill=tk.X, pady=5)
        
        
        viz_controls = ttk.LabelFrame(left_panel, text=" Visualization Types ", padding=15)
        viz_controls.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        
        canvas_frame = ttk.Frame(viz_controls)
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        
       
        chart_canvas = tk.Canvas(canvas_frame, bg=self.card_bg, highlightthickness=0)
        scrollbar = ttk.Scrollbar(canvas_frame, orient="vertical", command=chart_canvas.yview)
        scrollable_frame = ttk.Frame(chart_canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: chart_canvas.configure(scrollregion=chart_canvas.bbox("all"))
        )
        
        chart_canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        chart_canvas.configure(yscrollcommand=scrollbar.set)
        
        chart_types = [
            ("📈 Bar Chart", self.show_bar_chart, "Categorical vs Numerical"),
            ("🔵 Scatter Plot", self.show_scatter_plot, "Two numerical variables"),
            ("📊 Line Chart", self.show_line_chart, "Time series data"),
            ("📉 Histogram", self.show_histogram, "Distribution of data"),
            ("🔥 Heatmap", self.show_heatmap, "Correlation matrix"),
            ("🍩 Pie Chart", self.show_pie_chart, "Composition/proportions"),
            ("📊 Box Plot", self.show_box_plot, "Statistical distribution"),
            ("📈 Area Chart", self.show_area_chart, "Cumulative data")
        ]
        
        for i, (text, command, tooltip) in enumerate(chart_types):
            frame = ttk.Frame(scrollable_frame)
            frame.pack(fill=tk.X, padx=5, pady=3)
            
            btn = ttk.Button(frame, text=text, command=command, style='Secondary.TButton')
            btn.pack(fill=tk.X)
            
            ttk.Label(frame, text=tooltip, font=('Segoe UI', 8), foreground=self.text_light).pack(anchor=tk.W, padx=5)
        
        chart_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        
        paned_window.add(left_panel, weight=1)
        
        
        right_panel = ttk.Frame(paned_window)
        paned_window.add(right_panel, weight=4)  
        
        
        canvas_section = ttk.LabelFrame(right_panel, text=" Chart Display ", padding=10)
        canvas_section.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        
        canvas_controls = ttk.Frame(canvas_section)
        canvas_controls.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(canvas_controls, text="Chart Options:", font=('Segoe UI', 10, 'bold')).pack(side=tk.LEFT)
        
        ttk.Button(canvas_controls, text="🗑️ Clear", command=self.clear_canvas, style='Secondary.TButton').pack(side=tk.RIGHT, padx=5)
        ttk.Button(canvas_controls, text="🖼️ Save", command=self.save_chart, style='Secondary.TButton').pack(side=tk.RIGHT, padx=5)
        ttk.Button(canvas_controls, text="🔍 Zoom", command=self.add_toolbar, style='Secondary.TButton').pack(side=tk.RIGHT, padx=5)
        
        
        self.canvas_container = ttk.Frame(canvas_section)
        self.canvas_container.pack(fill=tk.BOTH, expand=True)
        
        
        self.placeholder_label = ttk.Label(self.canvas_container,
                                          text="👈 Select a chart type to display visualization\n\n📈 Charts will appear here with full interactive features",
                                          font=('Segoe UI', 12),
                                          foreground=self.text_light,
                                          justify=tk.CENTER)
        self.placeholder_label.pack(expand=True)
    
    def create_linear_regression_tab(self):
        """Enhanced linear regression tab with larger visualization area"""
        main_container = ttk.Frame(self.lr_frame)
        main_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        
        paned_window = ttk.PanedWindow(main_container, orient=tk.VERTICAL)
        paned_window.pack(fill=tk.BOTH, expand=True)
        
        
        top_panel = ttk.Frame(paned_window, height=200)
        top_panel.pack_propagate(False)
        
        
        data_section = ttk.LabelFrame(top_panel, text=" Data Management ", padding=15)
        data_section.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        
        source_frame = ttk.Frame(data_section)
        source_frame.pack(fill=tk.X, pady=10)
        
        sources = [
            ("🏠 California Housing", self.load_predefined_data, "Built-in dataset"),
            ("📁 Custom CSV", self.load_custom_regression_data, "Your own data"),
            ("📊 Generate Sample", self.generate_sample_data, "Random data")
        ]
        
        for i, (text, command, tooltip) in enumerate(sources):
            frame = ttk.Frame(source_frame)
            frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10) if i < 2 else 0)
            
            btn = ttk.Button(frame, text=text, command=command, style='TButton' if i == 0 else 'Secondary.TButton')
            btn.pack(fill=tk.BOTH, expand=True)
            
            ttk.Label(frame, text=tooltip, font=('Segoe UI', 8), foreground=self.text_light).pack()
        
        
        training_frame = ttk.Frame(data_section)
        training_frame.pack(fill=tk.X, pady=(15, 0))
        
        train_btn = ttk.Button(training_frame, text="🎯 Train Model", command=self.train_regression_model, style='TButton')
        train_btn.pack(side=tk.LEFT, padx=(0, 10))
        


        self.lr_label = ttk.Label(training_frame,
                                 text="Status: Ready to load data",
                                 foreground=self.text_light,
                                 font=('Segoe UI', 10))
        self.lr_label.pack(side=tk.LEFT)
        
        
        
        viz_controls_frame = ttk.Frame(data_section)
        viz_controls_frame.pack(fill=tk.X, pady=(15, 0))
        
        viz_options = [
            ("📈 Predictions", self.show_lr_predictions, "Actual vs Predicted"),
            ("📊 Residuals", self.show_residuals, "Error distribution"),
            ("⭐ Importance", self.show_feature_importance, "Feature coefficients"),
            ("📋 Metrics", self.show_metrics, "Performance metrics")
        ]
        
        for i, (text, command, tooltip) in enumerate(viz_options):
            frame = ttk.Frame(viz_controls_frame)
            frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10) if i < 3 else 0)
            
            btn = ttk.Button(frame, text=text, command=command, style='Secondary.TButton')
            btn.pack(fill=tk.BOTH, expand=True)
            
            ttk.Label(frame, text=tooltip, font=('Segoe UI', 8), foreground=self.text_light).pack()
        
        paned_window.add(top_panel, weight=1)
        
        
        
        bottom_panel = ttk.Frame(paned_window)
        paned_window.add(bottom_panel, weight=3)  
        
        
       
        canvas_section = ttk.LabelFrame(bottom_panel, text=" Model Analysis ", padding=10)
        canvas_section.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        
        canvas_controls = ttk.Frame(canvas_section)
        canvas_controls.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(canvas_controls, text="Analysis Options:", font=('Segoe UI', 10, 'bold')).pack(side=tk.LEFT)
        
        ttk.Button(canvas_controls, text="🗑️ Clear", command=self.clear_lr_canvas, style='Secondary.TButton').pack(side=tk.RIGHT, padx=5)
        ttk.Button(canvas_controls, text="🖼️ Save", command=self.save_lr_chart, style='Secondary.TButton').pack(side=tk.RIGHT, padx=5)
        ttk.Button(canvas_controls, text="🔍 Zoom", command=self.add_lr_toolbar, style='Secondary.TButton').pack(side=tk.RIGHT, padx=5)
        
        
        self.lr_canvas_container = ttk.Frame(canvas_section)
        self.lr_canvas_container.pack(fill=tk.BOTH, expand=True)
        
        
        self.lr_placeholder_label = ttk.Label(self.lr_canvas_container,
                                             text="👈 Load data and train model to see analysis\n\n📈 Model visualizations will appear here with full size",
                                             font=('Segoe UI', 12),
                                             foreground=self.text_light,
                                             justify=tk.CENTER)
        self.lr_placeholder_label.pack(expand=True)
    
    def add_toolbar(self):
        """Add matplotlib navigation toolbar to visualization tab"""
        if hasattr(self, 'current_figure') and hasattr(self, 'current_canvas'):
            
            for widget in self.canvas_container.winfo_children():
                if isinstance(widget, tk.Frame) and widget.winfo_children():
                    if isinstance(widget.winfo_children()[0], NavigationToolbar2Tk):
                        widget.destroy()
            
            
            toolbar_frame = ttk.Frame(self.canvas_container)
            toolbar_frame.pack(side=tk.BOTTOM, fill=tk.X)
            
            toolbar = NavigationToolbar2Tk(self.current_canvas, toolbar_frame)
            toolbar.update()
    
    def add_lr_toolbar(self):
        """Add matplotlib navigation toolbar to ML tab"""
        if hasattr(self, 'lr_current_figure') and hasattr(self, 'lr_current_canvas'):
            
            for widget in self.lr_canvas_container.winfo_children():
                if isinstance(widget, tk.Frame) and widget.winfo_children():
                    if isinstance(widget.winfo_children()[0], NavigationToolbar2Tk):
                        widget.destroy()
            
            
            toolbar_frame = ttk.Frame(self.lr_canvas_container)
            toolbar_frame.pack(side=tk.BOTTOM, fill=tk.X)
            
            toolbar = NavigationToolbar2Tk(self.lr_current_canvas, toolbar_frame)
            toolbar.update()
    
    def show_box_plot(self):
        """Show box plot"""
        if self.df is None:
            messagebox.showerror("Error", "Load data first")
            return
        
        self.clear_canvas()
        try:
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) == 0:
                messagebox.showerror("Error", "No numeric columns found")
                return
            
            fig = Figure(figsize=(10, 6), dpi=100)
            ax = fig.add_subplot(111)
            
            
            plot_data = [self.df[col].dropna() for col in numeric_cols[:5]]
            ax.boxplot(plot_data, labels=numeric_cols[:5])
            
            ax.set_title('Box Plot', fontweight='bold')
            ax.set_ylabel('Value')
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='x', rotation=45)
            
            fig.tight_layout()
            
            self.current_figure = fig
            self.current_canvas = FigureCanvasTkAgg(fig, self.canvas_container)
            self.current_canvas.draw()
            self.current_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create box plot: {str(e)}")
    
    def show_area_chart(self):
        """Show area chart"""
        if self.df is None:
            messagebox.showerror("Error", "Load data first")
            return
        
        self.clear_canvas()
        try:
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) == 0:
                messagebox.showerror("Error", "No numeric columns found")
                return
            
            fig = Figure(figsize=(10, 6), dpi=100)
            ax = fig.add_subplot(111)
            
            
            for i, col in enumerate(numeric_cols[:3]):
                ax.fill_between(self.df.index, 0, self.df[col], alpha=0.5, label=col)
            
            ax.set_title('Area Chart', fontweight='bold')
            ax.set_xlabel('Index')
            ax.set_ylabel('Value')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            fig.tight_layout()
            
            self.current_figure = fig
            self.current_canvas = FigureCanvasTkAgg(fig, self.canvas_container)
            self.current_canvas.draw()
            self.current_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create area chart: {str(e)}")
    
    def clear_canvas(self):
        """Clear visualization canvas"""
        for widget in self.canvas_container.winfo_children():
            widget.destroy()
    
    def clear_lr_canvas(self):
        """Clear ML visualization canvas"""
        for widget in self.lr_canvas_container.winfo_children():
            widget.destroy()
    
    def save_chart(self):
        """Save current chart as image"""
        if hasattr(self, 'current_figure'):
            file_path = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG files", "*.png"), ("PDF files", "*.pdf"), ("SVG files", "*.svg"), ("All files", "*.*")]
            )
            
            if file_path:
                try:
                    self.current_figure.savefig(file_path, dpi=300, bbox_inches='tight')
                    messagebox.showinfo("Success", f"Chart saved to:\n{file_path}")
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to save: {str(e)}")
        else:
            messagebox.showinfo("Save Chart", "No chart to save.")
    
    def save_lr_chart(self):
        """Save ML chart as image"""
        if hasattr(self, 'lr_current_figure'):
            file_path = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG files", "*.png"), ("PDF files", "*.pdf"), ("SVG files", "*.svg"), ("All files", "*.*")]
            )
            
            if file_path:
                try:
                    self.lr_current_figure.savefig(file_path, dpi=300, bbox_inches='tight')
                    messagebox.showinfo("Success", f"Chart saved to:\n{file_path}")
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to save: {str(e)}")
        else:
            messagebox.showinfo("Save Chart", "No chart to save.")
    
    

    
    def swap_matrices(self):
        """Swap matrix A and B"""
        if self.matrix_a is not None or self.matrix_b is not None:
            self.matrix_a, self.matrix_b = self.matrix_b, self.matrix_a
            
            if self.matrix_a is not None:
                self.label_a.config(text=f"✓ {self.matrix_a.shape}", foreground=self.success_color)
            else:
                self.label_a.config(text="Not loaded", foreground=self.warning_color)
            
            if self.matrix_b is not None:
                self.label_b.config(text=f"✓ {self.matrix_b.shape}", foreground=self.success_color)
            else:
                self.label_b.config(text="Not loaded", foreground=self.warning_color)
            
            self.display_result("Matrices swapped successfully!")
            self.status_label.config(text="Matrices swapped")
    
    def clear_matrices(self):
        """Clear all matrices"""
        self.matrix_a = None
        self.matrix_b = None
        self.label_a.config(text="Not loaded", foreground=self.warning_color)
        self.label_b.config(text="Not loaded", foreground=self.warning_color)
        self.display_result("All matrices cleared.")
        self.status_label.config(text="Matrices cleared")
    
    def show_line_chart(self):
        """Show line chart"""
        if self.df is None:
            messagebox.showerror("Error", "Load data first")
            return
        
        self.clear_canvas()
        try:
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) == 0:
                messagebox.showerror("Error", "No numeric columns found")
                return
            
            fig = Figure(figsize=(10, 6), dpi=100)
            ax = fig.add_subplot(111)
            
            for col in numeric_cols[:3]:
                ax.plot(self.df.index, self.df[col], label=col, marker='o', markersize=3)
            
            ax.set_title('Line Chart', fontweight='bold')
            ax.set_xlabel('Index')
            ax.set_ylabel('Value')
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            
            self.current_figure = fig
            self.current_canvas = FigureCanvasTkAgg(fig, self.canvas_container)
            self.current_canvas.draw()
            self.current_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create line chart: {str(e)}")
    
    def show_histogram(self):
        """Show histogram"""
        if self.df is None:
            messagebox.showerror("Error", "Load data first")
            return
        
        self.clear_canvas()
        try:
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) == 0:
                messagebox.showerror("Error", "No numeric columns found")
                return
            
            fig = Figure(figsize=(10, 6), dpi=100)
            ax = fig.add_subplot(111)
            
            for i, col in enumerate(numeric_cols[:3]):
                ax.hist(self.df[col].dropna(), alpha=0.6, label=col, bins=15)
            
            ax.set_title('Histogram', fontweight='bold')
            ax.set_xlabel('Value')
            ax.set_ylabel('Frequency')
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            
            self.current_figure = fig
            self.current_canvas = FigureCanvasTkAgg(fig, self.canvas_container)
            self.current_canvas.draw()
            self.current_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create histogram: {str(e)}")
    
    def show_pie_chart(self):
        """Show pie chart"""
        if self.df is None:
            messagebox.showerror("Error", "Load data first")
            return
        
        self.clear_canvas()
        try:
            categorical_cols = [col for col in self.df.columns if self.df[col].dtype == 'object']
            
            if not categorical_cols:
                messagebox.showerror("Error", "No categorical columns found for pie chart")
                return
            
            col = categorical_cols[0]
            value_counts = self.df[col].value_counts().head(8)
            
            fig = Figure(figsize=(8, 6), dpi=100)
            ax = fig.add_subplot(111)
            
            colors = plt.cm.Set3(np.linspace(0, 1, len(value_counts)))
            ax.pie(value_counts.values, labels=value_counts.index, autopct='%1.1f%%',
                  colors=colors, startangle=90)
            ax.set_title(f'Pie Chart: {col}', fontweight='bold')
            fig.tight_layout()
            
            self.current_figure = fig
            self.current_canvas = FigureCanvasTkAgg(fig, self.canvas_container)
            self.current_canvas.draw()
            self.current_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create pie chart: {str(e)}")
    
    def preview_data(self):
        """Preview loaded data"""
        if self.df is None:
            messagebox.showinfo("Data Preview", "No data loaded yet.")
            return
        
        preview_window = tk.Toplevel(self.root)
        preview_window.title("Data Preview")
        preview_window.geometry("800x500")
        
        text_widget = tk.Text(preview_window, wrap=tk.NONE)
        text_widget.pack(fill=tk.BOTH, expand=True)
        
        text_widget.insert(tk.END, f"Data Shape: {self.df.shape}\n")
        text_widget.insert(tk.END, f"Columns: {list(self.df.columns)}\n")
        text_widget.insert(tk.END, "="*80 + "\n\n")
        text_widget.insert(tk.END, "First 20 rows:\n")
        text_widget.insert(tk.END, str(self.df.head(20)))
        
        text_widget.config(state=tk.DISABLED)
    
    def generate_sample_data(self):
        """Generate sample regression data"""
        try:
            np.random.seed(42)
            n_samples = 100
            
            X = np.random.randn(n_samples, 3)
            coefficients = np.array([2.5, -1.2, 0.8])
            y = X @ coefficients + np.random.randn(n_samples) * 0.5
            
            self.X = pd.DataFrame(X, columns=['Feature_1', 'Feature_2', 'Feature_3'])
            self.y = pd.Series(y, name='Target')
            
            self.custom_data_loaded = True
            self.data_features = self.X.columns.tolist()
            
            status_text = f"✓ Sample Data Generated | {len(self.X)} samples, {len(self.X.columns)} features"
            self.lr_label.config(text=status_text, foreground=self.success_color)
            messagebox.showinfo("Success", "Sample data generated successfully!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate data: {str(e)}")
    
    def show_residuals(self):
        """Show residual plot"""
        if self.model is None:
            messagebox.showerror("Error", "Please train the model first")
            return
        
        self.clear_lr_canvas()
        try:
            residuals = self.y_test - self.predictions
            
            fig = Figure(figsize=(10, 6), dpi=100)
            ax = fig.add_subplot(111)
            
            ax.scatter(self.predictions, residuals, alpha=0.5, color=self.primary_color, s=10)
            ax.axhline(y=0, color='r', linestyle='--', lw=2)
            ax.set_title('Residual Plot', fontsize=12, fontweight='bold')
            ax.set_xlabel('Predicted Values')
            ax.set_ylabel('Residuals')
            ax.grid(True, alpha=0.3)
            
            fig.tight_layout()
            
            self.lr_current_figure = fig
            self.lr_current_canvas = FigureCanvasTkAgg(fig, self.lr_canvas_container)
            self.lr_current_canvas.draw()
            self.lr_current_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
        except Exception as e:
            messagebox.showerror("Error", str(e))
    
    def show_metrics(self):
        """Show performance metrics"""
        if self.model is None:
            messagebox.showerror("Error", "Please train the model first")
            return
        
        try:
            mse = mean_squared_error(self.y_test, self.predictions)
            mae = mean_absolute_error(self.y_test, self.predictions)
            r2 = r2_score(self.y_test, self.predictions)
            
            metrics_text = f"""
            ╔══════════════════════════════════╗
            ║      MODEL PERFORMANCE METRICS   ║
            ╠══════════════════════════════════╣
            ║ Mean Squared Error:  {mse:10.4f} ║
            ║ Mean Absolute Error: {mae:10.4f} ║
            ║ R² Score:            {r2:10.4f}  ║
            ╠══════════════════════════════════╣
            ║ Interpretation:                  ║
            ║ • R² = {r2:.2%} of variance      ║
            ║   explained by model             ║
            ╚══════════════════════════════════╝
            """
            
            self.clear_lr_canvas()
            
            text_widget = tk.Text(self.lr_canvas_container,
                                 font=("Consolas", 11),
                                 bg=self.card_bg,
                                 fg=self.text_color,
                                 borderwidth=0,
                                 padx=20,
                                 pady=20,
                                 state=tk.DISABLED)
            text_widget.pack(fill=tk.BOTH, expand=True)
            
            text_widget.config(state=tk.NORMAL)
            text_widget.delete(1.0, tk.END)
            text_widget.insert(1.0, metrics_text)
            text_widget.config(state=tk.DISABLED)
            
        except Exception as e:
            messagebox.showerror("Error", str(e))
    
    
    
    def apply_theme(self, theme_name):
        """Apply theme changes"""
        self.current_theme = theme_name
        style = ttk.Style()
        
        if theme_name == "corporate":
            self.bg_color = "#e8ecf1"
            self.card_bg = "#f0f4f9"
            self.primary_color = "#2d5f8d"
            self.text_color = "#2c3e50"
            self.text_light = "#5a6c7d"
        elif theme_name == "light":
            self.bg_color = "#f5f7fa"
            self.card_bg = "#ffffff"
            self.primary_color = "#3498db"
            self.text_color = "#34495e"
            self.text_light = "#7f8c8d"
        elif theme_name == "dark":
            self.bg_color = "#1a1a2e"
            self.card_bg = "#16213e"
            self.primary_color = "#0f3460"
            self.text_color = "#eaeaea"
            self.text_light = "#bdc3c7"
        
        self.root.configure(bg=self.bg_color)
        messagebox.showinfo("Theme", f"{theme_name.capitalize()} theme applied! 🎨")
    
    def apply_font(self, font_name):
        """Apply font changes"""
        self.current_font_name = font_name
        try:
            style = ttk.Style()
            style.configure('TButton', font=(font_name, 9, 'bold'))
            style.configure('TLabel', font=(font_name, 9))
            style.configure('Header.TLabel', font=(font_name, 13, 'bold'))
            style.configure('TLabelFrame.Label', font=(font_name, 11, 'bold'))
            
            if hasattr(self, 'result_text'):
                self.result_text.config(state=tk.NORMAL, font=(font_name, 9))
                self.result_text.config(state=tk.DISABLED)
            
            messagebox.showinfo("Font Applied", f"Font changed to {font_name}! 🔤")
        except Exception as e:
            messagebox.showerror("Error", f"Could not apply font: {str(e)}")
    
    def input_matrix_dialog(self, matrix_name="Matrix"):
        dialog = tk.Toplevel(self.root)
        dialog.title(f"Input {matrix_name}")
        dialog.geometry("550x650")
        dialog.minsize(500, 550)
        
        self.matrix_result = None
        
        button_frame = ttk.Frame(dialog)
        button_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=15, pady=15)
        
        content_frame = ttk.Frame(dialog)
        content_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        ttk.Label(content_frame, text=f"Enter {matrix_name} dimensions:", font=("Arial", 11, "bold")).pack(pady=10)
        
        ttk.Label(content_frame, text="Rows:").pack(anchor=tk.W)
        rows_entry = ttk.Entry(content_frame, width=30)
        rows_entry.pack(anchor=tk.W, pady=5)
        
        ttk.Label(content_frame, text="Columns:").pack(anchor=tk.W)
        cols_entry = ttk.Entry(content_frame, width=30)
        cols_entry.pack(anchor=tk.W, pady=5)
        
        ttk.Label(content_frame, text="Enter matrix elements:\n(one row per line)", font=("Arial", 9)).pack(anchor=tk.W, pady=(15, 5))
        
        text_widget = tk.Text(content_frame, height=12, width=50, font=("Courier", 9))
        text_widget.pack(fill=tk.BOTH, expand=True, pady=5)
        
        def parse_matrix():
            try:
                rows = int(rows_entry.get().strip())
                cols = int(cols_entry.get().strip())
                
                if rows <= 0 or cols <= 0:
                    messagebox.showerror("Error", "Rows and Columns must be positive")
                    return
                
                text = text_widget.get("1.0", tk.END).strip()
                elements = []
                for line in text.split('\n'):
                    if line.strip():
                        row = [float(x) for x in line.replace(',', ' ').split()]
                        elements.append(row)
                
                if len(elements) != rows:
                    messagebox.showerror("Error", f"Expected {rows} rows")
                    return
                
                if any(len(row) != cols for row in elements):
                    messagebox.showerror("Error", f"Each row must have {cols} columns")
                    return
                
                self.matrix_result = np.array(elements)
                messagebox.showinfo("Success", f"{matrix_name} saved!")
                dialog.destroy()
            except ValueError as e:
                messagebox.showerror("Error", f"Invalid format: {str(e)}")
        
        ttk.Button(button_frame, text="Submit", command=parse_matrix, width=25).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=dialog.destroy, width=25).pack(side=tk.LEFT, padx=5)
        
        dialog.transient(self.root)
        dialog.grab_set()
        self.root.wait_window(dialog)
        
        return self.matrix_result
    
    def input_matrix_a(self):
        self.matrix_a = self.input_matrix_dialog("Matrix A")
        if self.matrix_a is not None:
            self.label_a.config(text=f"✓ {self.matrix_a.shape}", foreground=self.success_color)
            self.display_result(f"Matrix A loaded: {self.matrix_a.shape}\n{self.matrix_a}")
            self.status_label.config(text="Matrix A loaded")
    
    def input_matrix_b(self):
        self.matrix_b = self.input_matrix_dialog("Matrix B")
        if self.matrix_b is not None:
            self.label_b.config(text=f"✓ {self.matrix_b.shape}", foreground=self.success_color)
            self.display_result(f"Matrix B loaded: {self.matrix_b.shape}\n{self.matrix_b}")
            self.status_label.config(text="Matrix B loaded")
    
    def display_result(self, result):
        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete("1.0", tk.END)
        self.result_text.insert("1.0", str(result))
        self.result_text.config(state=tk.DISABLED)
    
    def matrix_add(self):
        if self.matrix_a is None or self.matrix_b is None:
            messagebox.showerror("Error", "Both matrices needed")
            return
        if self.matrix_a.shape != self.matrix_b.shape:
            messagebox.showerror("Error", "Same dimensions required")
            return
        result = self.matrix_a + self.matrix_b
        self.display_result(f"A + B:\n{result}")
    
    def matrix_subtract(self):
        if self.matrix_a is None or self.matrix_b is None:
            messagebox.showerror("Error", "Both matrices needed")
            return
        if self.matrix_a.shape != self.matrix_b.shape:
            messagebox.showerror("Error", "Same dimensions required")
            return
        result = self.matrix_a - self.matrix_b
        self.display_result(f"A - B:\n{result}")
    
    def matrix_multiply(self):
        if self.matrix_a is None or self.matrix_b is None:
            messagebox.showerror("Error", "Both matrices needed")
            return
        if self.matrix_a.shape[1] != self.matrix_b.shape[0]:
            messagebox.showerror("Error", "Column of A must equal row of B")
            return
        result = np.dot(self.matrix_a, self.matrix_b)
        self.display_result(f"A × B:\n{result}")
    
    def transpose_a(self):
        if self.matrix_a is None:
            messagebox.showerror("Error", "Matrix A not loaded")
            return
        result = self.matrix_a.T
        self.display_result(f"Transpose of A:\n{result}")
    
    def transpose_b(self):
        if self.matrix_b is None:
            messagebox.showerror("Error", "Matrix B not loaded")
            return
        result = self.matrix_b.T
        self.display_result(f"Transpose of B:\n{result}")
    
    def determinant_a(self):
        if self.matrix_a is None:
            messagebox.showerror("Error", "Matrix A not loaded")
            return
        if self.matrix_a.shape[0] != self.matrix_a.shape[1]:
            messagebox.showerror("Error", "Must be square")
            return
        try:
            det = np.linalg.det(self.matrix_a)
            self.display_result(f"Determinant of A: {det}")
        except Exception as e:
            messagebox.showerror("Error", str(e))
    
    def determinant_b(self):
        if self.matrix_b is None:
            messagebox.showerror("Error", "Matrix B not loaded")
            return
        if self.matrix_b.shape[0] != self.matrix_b.shape[1]:
            messagebox.showerror("Error", "Must be square")
            return
        try:
            det = np.linalg.det(self.matrix_b)
            self.display_result(f"Determinant of B: {det}")
        except Exception as e:
            messagebox.showerror("Error", str(e))
    
    def load_sample_data(self):
        try:
            sample_data = {
                'Product': ['Laptop', 'Mouse', 'Monitor', 'Keyboard', 'Webcam', 'Headset'],
                'Category': ['Electronics', 'Accessories', 'Electronics', 'Accessories', 'Accessories', 'Accessories'],
                'Units_Sold': [50, 150, 80, 120, 90, 110],
                'Revenue': [60000, 3750, 24000, 9000, 4500, 8800]
            }
            self.df = pd.DataFrame(sample_data)
            
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            avg_info = "\n".join([f"{col}: {self.df[col].mean():.2f}" for col in numeric_cols])
            
            self.data_label.config(text=f"✓ {len(self.df)} products loaded\nAverage:\n{avg_info}", 
                                  foreground=self.success_color)
            self.status_label.config(text="Sample data ready")
            messagebox.showinfo("Success", f"Sample data loaded!\n\nAverages:\n{avg_info}")
        except Exception as e:
            messagebox.showerror("Error", str(e))
    
    def load_csv(self):
        file = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file:
            try:
                self.df = pd.read_csv(file)
                
                numeric_cols = self.df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    avg_info = "\n".join([f"{col}: {self.df[col].mean():.2f}" for col in numeric_cols])
                    self.data_label.config(text=f"✓ {len(self.df)} rows\nAverage:\n{avg_info}", 
                                          foreground=self.success_color)
                    messagebox.showinfo("Success", f"CSV loaded successfully!\n\nAverages:\n{avg_info}")
                else:
                    self.data_label.config(text=f"✓ {len(self.df)} rows (no numeric columns)", 
                                          foreground=self.success_color)
                    messagebox.showinfo("Success", "CSV loaded successfully (no numeric columns)")
                
                self.status_label.config(text="CSV loaded")
            except Exception as e:
                messagebox.showerror("Error", str(e))
    
    def show_bar_chart(self):
        if self.df is None:
            messagebox.showerror("Error", "Load data first")
            return
        self.clear_canvas()
        try:
            categorical_col = None
            numeric_col = None
            
            for col in self.df.columns:
                if categorical_col is None and self.df[col].dtype == 'object':
                    categorical_col = col
                if numeric_col is None and self.df[col].dtype in [np.number, np.int64, np.float64]:
                    numeric_col = col
            
            if categorical_col is None or numeric_col is None:
                messagebox.showerror("Error", "CSV must have categorical and numeric columns")
                return
            
            fig = Figure(figsize=(10, 6), dpi=100)
            ax = fig.add_subplot(111)
            ax.bar(self.df[categorical_col], self.df[numeric_col], color=self.primary_color)
            ax.set_title(f'{numeric_col} by {categorical_col}', fontweight='bold')
            ax.set_ylabel(numeric_col)
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            
            self.current_figure = fig
            self.current_canvas = FigureCanvasTkAgg(fig, self.canvas_container)
            self.current_canvas.draw()
            self.current_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
        except Exception as e:
            messagebox.showerror("Error", str(e))
    
    def show_scatter_plot(self):
        if self.df is None:
            messagebox.showerror("Error", "Load data first")
            return
        self.clear_canvas()
        try:
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) < 2:
                messagebox.showerror("Error", "CSV needs at least 2 numeric columns")
                return
            
            x_col = numeric_cols[0]
            y_col = numeric_cols[1]
            
            categorical_col = None
            for col in self.df.columns:
                if self.df[col].dtype == 'object':
                    categorical_col = col
                    break
            
            fig = Figure(figsize=(10, 6), dpi=100)
            ax = fig.add_subplot(111)
            
            if categorical_col is not None:
                for category in self.df[categorical_col].unique():
                    cat_data = self.df[self.df[categorical_col] == category]
                    ax.scatter(cat_data[x_col], cat_data[y_col], label=str(category), s=100, alpha=0.6)
                ax.legend()
            else:
                ax.scatter(self.df[x_col], self.df[y_col], s=100, alpha=0.6, color=self.primary_color)
            
            ax.set_title(f'{y_col} vs {x_col}', fontweight='bold')
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            
            self.current_figure = fig
            self.current_canvas = FigureCanvasTkAgg(fig, self.canvas_container)
            self.current_canvas.draw()
            self.current_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
        except Exception as e:
            messagebox.showerror("Error", str(e))
    
    def show_heatmap(self):
        if self.df is None:
            messagebox.showerror("Error", "Load data first")
            return
        self.clear_canvas()
        try:
            fig = Figure(figsize=(10, 6), dpi=100)
            ax = fig.add_subplot(111)
            
            numeric_df = self.df.select_dtypes(include=[np.number])
            corr = numeric_df.corr()
            
            im = ax.imshow(corr, cmap='Blues', aspect='auto')
            ax.set_xticks(range(len(corr.columns)))
            ax.set_yticks(range(len(corr.columns)))
            ax.set_xticklabels(corr.columns, rotation=45, ha='right')
            ax.set_yticklabels(corr.columns)
            ax.set_title('Correlation Heatmap', fontweight='bold')
            
            for i in range(len(corr)):
                for j in range(len(corr)):
                    ax.text(j, i, f'{corr.iloc[i, j]:.2f}', ha='center', va='center', 
                           color='white' if abs(corr.iloc[i, j]) > 0.5 else 'black', fontsize=9)
            
            fig.colorbar(im, ax=ax)
            fig.tight_layout()
            
            self.current_figure = fig
            self.current_canvas = FigureCanvasTkAgg(fig, self.canvas_container)
            self.current_canvas.draw()
            self.current_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
        except Exception as e:
            messagebox.showerror("Error", str(e))
    
    def load_custom_regression_data(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            title="Select CSV file for regression"
        )
        
        if not file_path:
            return
        
        try:
            dialog = tk.Toplevel(self.root)
            dialog.title("Select Target Column")
            dialog.geometry("400x250")
            dialog.transient(self.root)
            dialog.grab_set()
            
            df = pd.read_csv(file_path)
            self.target_column = tk.StringVar()
            
            ttk.Label(dialog, text="Select the target column (dependent variable):", 
                     font=('Segoe UI', 10, 'bold')).pack(pady=15)
            
            combo = ttk.Combobox(dialog, textvariable=self.target_column, 
                                values=df.columns.tolist(), state='readonly', width=30)
            combo.pack(pady=10)
            
            def confirm_selection():
                if not self.target_column.get():
                    messagebox.showerror("Error", "Please select a target column")
                    return
                
                target_col = self.target_column.get()
                try:
                    if target_col not in df.columns:
                        raise ValueError(f"Column '{target_col}' not found")
                    
                    df_clean = df.dropna()
                    self.X = df_clean.drop(target_col, axis=1)
                    self.y = df_clean[target_col]
                    
                    self.X = self.X.select_dtypes(include=[np.number])
                    
                    if len(self.X.columns) == 0:
                        raise ValueError("No numeric features found in the data")
                    
                    self.custom_data_loaded = True
                    self.data_features = self.X.columns.tolist()
                    
                    status_text = f"✓ Custom Data Loaded | {len(self.X)} samples, {len(self.X.columns)} features"
                    self.lr_label.config(text=status_text, foreground=self.success_color)
                    messagebox.showinfo("Success", f"Custom data loaded successfully!\nSamples: {len(self.X)}\nFeatures: {len(self.X.columns)}")
                    dialog.destroy()
                    
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to load data: {str(e)}")
            
            ttk.Button(dialog, text="Confirm", command=confirm_selection).pack(pady=10)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file: {str(e)}")
    
    def load_predefined_data(self):
        try:
            self.lr_label.config(text="Loading predefined data...", foreground=self.accent_color)
            self.root.update()
            
            data = fetch_california_housing()
            df = pd.DataFrame(data.data, columns=data.feature_names)
            df['Price'] = data.target
            
            self.X = df.drop('Price', axis=1)
            self.y = df['Price']
            self.custom_data_loaded = False
            self.data_features = data.feature_names
            
            status_text = f"✓ Predefined Data Loaded | {len(self.X)} samples, {len(self.X.columns)} features"
            self.lr_label.config(text=status_text, foreground=self.success_color)
            messagebox.showinfo("Success", "Predefined California Housing data loaded!\nClick 'Train Model' to train.")
            
        except Exception as e:
            self.lr_label.config(text=f"Error: {str(e)}", foreground=self.accent_color)
            messagebox.showerror("Error", f"Failed to load data: {str(e)}")
    
    def train_regression_model(self):
        if self.X is None or self.y is None:
            messagebox.showerror("Error", "Please load data first (Predefined or Custom)")
            return
        
        try:
            self.lr_label.config(text="Training model... Please wait", foreground=self.accent_color)
            self.root.update()
            
            self.feature_names = self.X.columns.tolist()
            
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X, self.y, test_size=0.2, random_state=42
            )
            
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(self.X_train)
            X_test_scaled = self.scaler.transform(self.X_test)
            
            self.model = LinearRegression()
            self.model.fit(X_train_scaled, self.y_train)
            
            self.predictions = self.model.predict(X_test_scaled)
            
            mse = mean_squared_error(self.y_test, self.predictions)
            mae = mean_absolute_error(self.y_test, self.predictions)
            r2 = r2_score(self.y_test, self.predictions)
            
            status_text = f"✓ Model Trained | MAE: {mae:,.4f} | MSE: {mse:.4f} | R²: {r2:.4f}"
            self.lr_label.config(text=status_text, foreground=self.success_color)
            messagebox.showinfo("Success", f"Model trained successfully!\nR² Score: {r2:.4f}\nMAE: {mae:,.4f}")
            
        except Exception as e:
            self.lr_label.config(text=f"Error: {str(e)}", foreground=self.accent_color)
            messagebox.showerror("Error", f"Failed to train model: {str(e)}")
    
    def show_lr_predictions(self):
        if self.model is None:
            messagebox.showerror("Error", "Please train the model first")
            return
        
        self.clear_lr_canvas()
        try:
            fig = Figure(figsize=(10, 6), dpi=100)
            ax = fig.add_subplot(111)
            
            ax.scatter(self.y_test, self.predictions, alpha=0.5, color=self.primary_color, s=10)
            ax.plot([self.y_test.min(), self.y_test.max()], 
                   [self.y_test.min(), self.y_test.max()], '--r', lw=2)
            
            data_type = "Custom Data" if self.custom_data_loaded else "Predefined Data (Housing)"
            ax.set_title(f'Actual vs Predicted Values ({data_type})', fontsize=12, fontweight='bold')
            ax.set_xlabel('Actual Values')
            ax.set_ylabel('Predicted Values')
            ax.grid(True, alpha=0.3)
            
            fig.tight_layout()
            
            self.lr_current_figure = fig
            self.lr_current_canvas = FigureCanvasTkAgg(fig, self.lr_canvas_container)
            self.lr_current_canvas.draw()
            self.lr_current_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
        except Exception as e:
            messagebox.showerror("Error", str(e))
    
    def show_feature_importance(self):
        if self.model is None:
            messagebox.showerror("Error", "Please train the model first")
            return
        
        self.clear_lr_canvas()
        try:
            fig = Figure(figsize=(10, 6), dpi=100)
            ax = fig.add_subplot(111)
            
            if not hasattr(self, 'feature_names') or self.feature_names is None:
                messagebox.showerror("Error", "Feature names not found. Please train the model again.")
                return
            
            if len(self.model.coef_) != len(self.feature_names):
                messagebox.showerror("Error", f"Feature mismatch: Model has {len(self.model.coef_)} coefficients but {len(self.feature_names)} feature names. Please train the model again.")
                return
            
            coefficients = pd.Series(self.model.coef_, index=self.feature_names).sort_values()
            coefficients.plot(kind='barh', ax=ax, color=self.primary_color)
            ax.set_title('Feature Importance (Coefficients)', fontsize=12, fontweight='bold')
            ax.set_xlabel('Coefficient Value')
            ax.grid(True, alpha=0.3, axis='x')
            
            fig.tight_layout()
            
            self.lr_current_figure = fig
            self.lr_current_canvas = FigureCanvasTkAgg(fig, self.lr_canvas_container)
            self.lr_current_canvas.draw()
            self.lr_current_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
        except Exception as e:
            messagebox.showerror("Error", str(e))
    
    def show_about(self):
        about_text = """📊 DATA ANALYSIS & VISUALIZATION TOOL
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DESCRIPTION:
A comprehensive professional data analysis platform combining 
matrix operations, data visualization, and machine learning 
capabilities in a single, user-friendly interface.

KEY FEATURES:
✓ Matrix Operations - Add, subtract, multiply, transpose & determinants
✓ Data Visualization - Bar charts, scatter plots, heatmaps
✓ Linear Regression - Train with predefined or custom data
✓ Customization - Theme switching, Font selection
✓ Professional Design - Neumorphic UI with responsive layout

SUPPORTED THEMES:
• Corporate (Default) - Professional blue color scheme
• Light - Clean and bright appearance
• Dark - Easy on the eyes

AVAILABLE FONTS:
Segoe UI, Consolas, Arial, Times New Roman, 
Courier New, Georgia, Calibri, Verdana

MACHINE LEARNING:
✓ Train models on California Housing data
✓ Load custom datasets from CSV files
✓ Automatic feature scaling and validation
✓ Real-time performance metrics (MAE, MSE, R²)

Built with Python, scikit-learn, pandas, and matplotlib.

                        Developed by : Vikash Ramdarash Chaurasiya"""
        messagebox.showinfo("About Application", about_text)
    
    def show_guide(self):
        guide_text = """📖 HOW TO USE - QUICK GUIDE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🧮 MATRIX OPERATIONS:
1. Click "Matrix Ops" tab
2. Input Matrix A - Click "Input" → Enter dimensions → Enter values
3. Input Matrix B - Click "Input" → Enter dimensions → Enter values
4. Select operation: Add, Subtract, Multiply, Transpose, Determinant
5. View results in the output area

📊 DATA VISUALIZATION:
1. Click "Data Viz" tab
2. Load data:
   • Click "📤 Load CSV" to load your own CSV file
   • Click "📊 Sample Data" to use built-in sample data
3. Choose visualization:
   • 📈 Bar Chart - Shows categorical vs numeric data
   • 🔵 Scatter Plot - Shows relationship between two variables
   • 🔥 Heatmap - Shows correlation between all numeric columns

🤖 LINEAR REGRESSION:
1. Click "Regression" tab
2. Load data:
   • 🏠 Predefined Data - Uses California Housing dataset
   • 📂 Load Custom Data - Select your CSV file and target column
3. Click 🎯 "Train Model" - Trains the regression model
4. Analyze results:
   • 📈 Show Predictions - View actual vs predicted values
   • ⭐ Feature Importance - See which features matter most

🎨 CUSTOMIZATION:
1. Left sidebar → Theme section → Choose theme
2. Left sidebar → Font section → Choose font

💡 TIPS:
• All datasets must be in CSV format
• For custom data, numeric columns are automatically detected
• Hover over buttons for descriptions
• Use "About" for feature information"""
        messagebox.showinfo("User Guide", guide_text)
    
    def train_housing_model(self):
        """Legacy function: Redirect to load_predefined_data"""
        self.load_predefined_data()

if __name__ == "__main__":
    root = tk.Tk()
    app = MatrixGUI(root)
    root.mainloop()