import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
import requests
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# Configuración visual global
plt.style.use('ggplot')
pd.options.display.float_format = '{:,.0f}'.format

class TesisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("SICP - Sistema de Inteligencia de Contratación Pública")
        # Aumentamos un poco el tamaño de la ventana para que todo quepa mejor
        self.root.geometry("1200x800")
        
        # Variables Globales
        self.df = None
        self.df_ent = None
        self.kmeans = None
        self.scaler = None
        
        # Estilos
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TNotebook", tabposition='n')
        style.configure("TNotebook.Tab", font=('Arial', 11, 'bold'), padding=[10, 5])
        
        # --- ESTRUCTURA DE PESTAÑAS ---
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        self.tab1 = ttk.Frame(self.notebook) # Carga y EDA
        self.tab2 = ttk.Frame(self.notebook) # Clustering
        self.tab3 = ttk.Frame(self.notebook) # Auditoría
        
        self.notebook.add(self.tab1, text=" 1. Datos y EDA ")
        self.notebook.add(self.tab2, text=" 2. Segmentación (K-Means) ")
        self.notebook.add(self.tab3, text=" 3. Auditoría y Consulta ")
        
        self._init_tab1()
        self._init_tab2()
        self._init_tab3()

    # ================= PESTAÑA 1: DATOS Y EDA =================
    def _init_tab1(self):
        frame_top = tk.Frame(self.tab1, height=60, bg="#f0f0f0")
        frame_top.pack(fill='x', padx=5, pady=5)
        
        btn_load = tk.Button(frame_top, text=" CARGAR DATOS SECOP II", 
                             command=self.cargar_datos, bg="#00695c", fg="white", font=("Arial", 11, "bold"))
        btn_load.pack(side='left', padx=15, pady=10)
        
        self.lbl_status = tk.Label(frame_top, text="Estado: Esperando descarga...", bg="#f0f0f0", fg="#555", font=("Arial", 10))
        self.lbl_status.pack(side='left', padx=10)
        
        # Área de Gráficos
        self.frame_plots = tk.Frame(self.tab1)
        self.frame_plots.pack(fill='both', expand=True, pady=10)

    def cargar_datos(self):
        self.lbl_status.config(text=" Descargando dataset... (Esto puede tardar unos segundos)", fg="#ff8f00")
        self.root.update()
        
        try:
            archivo = "secop_auditoria.csv"
            # Pedimos 40k registros para tener buena muestra
            url = "https://www.datos.gov.co/resource/p6dx-8zbt.csv?$limit=40000"
            
            if not os.path.exists(archivo):
                r = requests.get(url)
                with open(archivo, 'wb') as f: f.write(r.content)
            
            self.df = pd.read_csv(archivo, low_memory=False)
            
            # Limpieza y Estandarización
            cols = [c.lower() for c in self.df.columns]
            self.df.columns = cols
            
            self.col_val = next((c for c in cols if 'valor' in c or 'cuantia' in c), 'valor_total_adjudicacion')
            self.col_mod = next((c for c in cols if 'modalidad' in c), 'modalidad_de_contratacion')
            self.col_ent = next((c for c in cols if 'entidad' in c and 'nit' not in c), 'entidad')
            self.col_nit = next((c for c in cols if 'nit' in c), 'nit_entidad')

            # Limpieza Valores
            if self.df[self.col_val].dtype == 'O':
                self.df[self.col_val] = self.df[self.col_val].astype(str).str.replace(r'[$,]', '', regex=True)
            self.df[self.col_val] = pd.to_numeric(self.df[self.col_val], errors='coerce')
            self.df = self.df[self.df[self.col_val] > 0]
            
            self.mostrar_eda()
            self.lbl_status.config(text=f" Dataset Listo: {len(self.df)} registros cargados.", fg="green")
            
        except Exception as e:
            messagebox.showerror("Error", f"Fallo crítico: {str(e)}")
            self.lbl_status.config(text=" Error en carga", fg="red")

    def mostrar_eda(self):
        for widget in self.frame_plots.winfo_children(): widget.destroy()
        
        # Aumentamos el tamaño de la figura
        fig, axs = plt.subplots(1, 2, figsize=(12, 5), dpi=100)
        
        # --- CORRECCIÓN 1: GRÁFICO DE BARRAS HORIZONTALES (barh) ---
        # Esto evita que los nombres se corten abajo
        top_mod = self.df[self.col_mod].value_counts().head(5).sort_values(ascending=True) # Ordenar para que el mayor salga arriba
        
        axs[0].barh(top_mod.index, top_mod.values, color='#00897b') # barh = Horizontal
        axs[0].set_title("Top 5 Modalidades (Más frecuentes)")
        axs[0].set_xlabel("Cantidad de Contratos")
        # Ajustamos márgenes para que quepan los nombres largos
        
        # EDA 2: Histograma
        sns.histplot(self.df[self.col_val], bins=30, log_scale=True, ax=axs[1], color='#5e35b1')
        axs[1].set_title("Distribución de Presupuesto (Escala Log)")
        axs[1].set_xlabel("Valor del Contrato (COP)")
        axs[1].set_ylabel("Frecuencia")
        
        # Ajuste automático para que no se monten los textos
        plt.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, master=self.frame_plots)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True, padx=10, pady=10)

    # ================= PESTAÑA 2: CLUSTERING =================
    def _init_tab2(self):
        frame_ctrl = tk.Frame(self.tab2, bg="#eceff1")
        frame_ctrl.pack(fill='x', padx=10, pady=10)
        
        btn_run = tk.Button(frame_ctrl, text=" EJECUTAR MODELO K-MEANS", 
                            command=self.ejecutar_clustering, bg="#c62828", fg="white", font=("Arial", 11, "bold"))
        btn_run.pack(pady=10)
        
        frame_body = tk.Frame(self.tab2)
        frame_body.pack(fill='both', expand=True)
        
        self.frame_cluster_plot = tk.Frame(frame_body)
        self.frame_cluster_plot.pack(side='left', fill='both', expand=True)
        
        # Texto un poco más grande aquí también
        self.txt_resultados = tk.Text(frame_body, width=45, height=20, font=("Consolas", 11))
        self.txt_resultados.pack(side='right', fill='y', padx=10, pady=10)

    def ejecutar_clustering(self):
        if self.df is None:
            messagebox.showwarning("Alerta", "Primero carga los datos en la Pestaña 1")
            return
            
        self.df['es_directa'] = self.df[self.col_mod].astype(str).apply(lambda x: 1 if 'Directa' in x or 'directa' in x else 0)
        
        self.df_ent = self.df.groupby(self.col_nit).agg({
            self.col_ent: 'first',
            self.col_val: ['sum', 'mean'],
            self.col_nit: 'count',
            'es_directa': 'mean'
        }).reset_index()
        
        self.df_ent.columns = ['nit', 'nombre', 'monto_total', 'monto_promedio', 'num_contratos', 'pct_directa']
        self.df_ent = self.df_ent[self.df_ent['monto_total'] > 0]
        
        # K-Means
        X = self.df_ent[['monto_total', 'num_contratos', 'pct_directa']]
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        self.kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        self.df_ent['cluster'] = self.kmeans.fit_predict(X_scaled)
        
        # Visualización
        for widget in self.frame_cluster_plot.winfo_children(): widget.destroy()
        
        fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
        scatter = ax.scatter(self.df_ent['pct_directa'], self.df_ent['monto_total'], 
                             c=self.df_ent['cluster'], cmap='viridis', alpha=0.7)
        ax.set_yscale('log')
        ax.set_title("Mapa de Segmentación (Clusters)")
        ax.set_xlabel("% Contratación Directa")
        ax.set_ylabel("Monto Total (Log)")
        plt.colorbar(scatter, ax=ax, label='Perfil')
        plt.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, master=self.frame_cluster_plot)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True, padx=10, pady=10)
        
        resumen = self.df_ent.groupby('cluster')[['monto_total', 'num_contratos', 'pct_directa']].mean()
        self.txt_resultados.delete(1.0, tk.END)
        self.txt_resultados.insert(tk.END, ">>> CENTROIDES DE PERFILES:\n\n")
        self.txt_resultados.insert(tk.END, resumen.to_string())
        self.txt_resultados.insert(tk.END, "\n\nINTERPRETACIÓN:\nCluster 0: Pequeños Competitivos\nCluster 1: Operadores Masivos\nCluster 2: ALTO RIESGO (Directa)\nCluster 3: Ejecutores Estratégicos")

    # ================= PESTAÑA 3: AUDITORÍA (CORREGIDA) =================
    def _init_tab3(self):
        frame_search = tk.Frame(self.tab3, bg="#e3f2fd", pady=20)
        frame_search.pack(fill='x')
        
        tk.Label(frame_search, text="BUSCAR ENTIDAD:", bg="#e3f2fd", font=("Arial", 12, "bold")).pack(side='left', padx=20)
        
        self.ent_search = tk.Entry(frame_search, width=30, font=("Arial", 12)) # Campo de texto más grande
        self.ent_search.pack(side='left', padx=10)
        
        btn_search = tk.Button(frame_search, text="🔍 AUDITAR AHORA", command=self.buscar_entidad, 
                               bg="#1565c0", fg="white", font=("Arial", 11, "bold"))
        btn_search.pack(side='left', padx=10)
        
        self.lbl_result_title = tk.Label(self.tab3, text="RESULTADOS DEL ANÁLISIS FORENSE", font=("Arial", 14, "bold"), fg="#333", pady=10)
        self.lbl_result_title.pack()
        
        # --- CORRECCIÓN 2: FUENTE MÁS GRANDE EN RESULTADOS ---
        # Cambiamos el tamaño de letra de 10 a 13
        self.txt_audit = tk.Text(self.tab3, height=20, width=90, font=("Consolas", 13), bg="#fafafa", padx=10, pady=10)
        
        # Añadimos Scrollbar por si el texto es muy largo
        scrollbar = tk.Scrollbar(self.tab3, command=self.txt_audit.yview)
        self.txt_audit.config(yscrollcommand=scrollbar.set)
        
        scrollbar.pack(side='right', fill='y')
        self.txt_audit.pack(pady=10, fill='both', expand=True, padx=20)

    def buscar_entidad(self):
        if self.df_ent is None:
            messagebox.showwarning("Error", "Primero debes ejecutar el Clustering (Pestaña 2).")
            return
            
        query = self.ent_search.get().upper()
        if not query: return
        
        mask = self.df_ent['nombre'].astype(str).str.contains(query, case=False) | \
               self.df_ent['nit'].astype(str).str.contains(query)
        
        resultados = self.df_ent[mask]
        
        self.txt_audit.delete(1.0, tk.END)
        
        if resultados.empty:
            self.txt_audit.insert(tk.END, f" No se encontraron entidades con el nombre/NIT: '{query}'.\n")
        else
            self.txt_audit.insert(tk.END, f" SE ENCONTRARON {len(resultados)} COINCIDENCIAS:\n")
            self.txt_audit.insert(tk.END, "="*60 + "\n")
            
            # Ajusta este diccionario según lo que salga en tu Pestaña 2
            perfiles_dict = {
                0: "PERFIL 0: BAJO RIESGO / COMPETITIVO",
                1: "PERFIL 1: OPERADOR LOGÍSTICO MASIVO",
                2: "PERFIL 2: ALTO RIESGO (CONTRATACIÓN A DEDO)",
                3: "PERFIL 3: GRAN EJECUTOR PRESUPUESTAL"
            }
            
            for index, row in resultados.head(10).iterrows():
                cluster_id = int(row['cluster'])
                interpretacion = perfiles_dict.get(cluster_id, "Desconocido")
                
                # Lógica de colores simulada con texto
                alerta = " ¡ALERTA MÁXIMA!" if row['pct_directa'] >= 0.95 else " Normal"
                
                info = f"""
 ENTIDAD: {row['nombre']}
 NIT: {row['nit']}
--------------------------------------------------
 Monto Total:       $ {row['monto_total']:,.0f}
 Total Contratos:   {row['num_contratos']}
 % Directa:         {row['pct_directa']*100:.1f}%
--------------------------------------------------
  CLASIFICACIÓN:     {interpretacion}
 ESTADO AUDITORÍA:  {alerta}
\n"""
                self.txt_audit.insert(tk.END, info)
                self.txt_audit.insert(tk.END, "-"*60 + "\n")

if __name__ == "__main__":
    root = tk.Tk()
    app = TesisApp(root)
    root.mainloop()
