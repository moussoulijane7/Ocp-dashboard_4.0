import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import io
import base64

# Configuration de la page
st.set_page_config(
    page_title="Dashboard Paramètres Diagnostics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Augmenter la limite de cellules pour Pandas Styler
pd.set_option("styler.render.max_elements", 500000)  # Ajusté pour gérer 322810 cellules

# Fonctions utilitaires
def detect_outliers(df, group_cols, value_col):
    """Détecte les valeurs aberrantes en utilisant la méthode IQR."""
    result_df = df.copy()
    result_df['Outlier'] = False
    
    for name, group in df.groupby(group_cols):
        Q1 = group[value_col].quantile(0.25)
        Q3 = group[value_col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outlier_indices = group[(group[value_col] < lower_bound) | (group[value_col] > upper_bound)].index
        result_df.loc[outlier_indices, 'Outlier'] = True
    
    return result_df

def get_download_link(df, filename, text):
    """Génère un lien pour télécharger un DataFrame au format CSV."""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href

# Titre du dashboard
st.title("📊 Dashboard d'analyse des paramètres diagnostics--Mine BA")

# Sidebar pour le chargement de fichier
st.sidebar.header("Chargement des données")

# Option de chargement
upload_option = st.sidebar.radio(
    "Comment souhaitez-vous charger vos données ?",
    ("Télécharger un fichier Excel", "Utiliser un exemple de données")
)

# Dataframe global
df = None

if upload_option == "Télécharger un fichier Excel":
    uploaded_file = st.sidebar.file_uploader("Choisissez un fichier Excel", type=["xlsx", "xls"])
    
    if uploaded_file is not None:
        try:
            # Afficher une liste des feuilles disponibles dans le fichier Excel
            xls = pd.ExcelFile(uploaded_file)
            sheet_name = st.sidebar.selectbox("Sélectionnez une feuille", xls.sheet_names)
            
            # Chargement des données
            df = pd.read_excel(uploaded_file, sheet_name=sheet_name)
            
            # Vérifier si les colonnes nécessaires existent
            required_cols = ['Engin', 'Paramètres Diagnostic', 'Valeur moyenne']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                st.sidebar.error(f"Colonnes manquantes: {', '.join(missing_cols)}")
                st.sidebar.info("Veuillez mapper vos colonnes aux noms attendus:")
                
                col_mapping = {}
                for req_col in required_cols:
                    if req_col in df.columns:
                        col_mapping[req_col] = req_col
                    else:
                        col_mapping[req_col] = st.sidebar.selectbox(
                            f"Colonne pour '{req_col}':",
                            [""] + list(df.columns)
                        )
                
                # Appliquer le mapping s'il est complet
                if all(col_mapping.values()):
                    # Créer une copie avec les noms de colonnes corrects
                    df_mapped = df.copy()
                    for new_col, old_col in col_mapping.items():
                        df_mapped[new_col] = df[old_col]
                    df = df_mapped[required_cols + ['Heure'] if 'Heure' in df_mapped.columns else required_cols]
                    
                    if 'Heure' not in df.columns:
                        st.sidebar.warning("Colonne 'Heure' non trouvée. Certaines fonctionnalités seront limitées.")
                else:
                    st.stop()
            
            # Convertir la colonne 'Heure' en datetime si elle existe
            if 'Heure' in df.columns:
                try:
                    df['Heure'] = pd.to_datetime(df['Heure'], errors='coerce', dayfirst=True)
                    if df['Heure'].isna().all():
                        df['Heure'] = pd.to_datetime(df['Heure'], errors='coerce', dayfirst=False)
                except Exception as e:
                    st.sidebar.warning(f"Erreur lors de la conversion des dates: {e}")
            
            # Ajouter une option pour filtrer sur 'Fonctionnel?' si la colonne existe
            if 'Fonctionnel?' in df.columns:
                filter_functional = st.sidebar.checkbox("Filtrer sur 'Fonctionnel? = Oui'", value=True)
                if filter_functional:
                    df = df[df['Fonctionnel?'] == 'Oui']
        
        except Exception as e:
            st.sidebar.error(f"Erreur lors du chargement du fichier: {e}")
            st.stop()
else:
    # Données d'exemple
    st.sidebar.info("Utilisation de données d'exemple générées...")
    
    # Générer des données d'exemple
    np.random.seed(42)
    engins = ['Engin_A', 'Engin_B', 'Engin_C', 'Engin_D']
    parametres = ['Température', 'Pression', 'Vibration', 'Débit', 'Niveau']
    
    # Créer un dataframe vide
    data = []
    
    # Date de départ
    start_date = datetime.now() - timedelta(days=30)
    
    # Générer des valeurs pour chaque engin et paramètre
    for engin in engins:
        for param in parametres:
            # Définir une valeur de base et une tendance pour chaque combinaison
            base_value = np.random.uniform(10, 100)
            trend = np.random.choice([-0.1, 0, 0.1, 0.2])
            
            # Générer des mesures pour les 30 derniers jours
            for day in range(30):
                for _ in range(np.random.randint(1, 4)):  # 1-3 mesures par jour
                    date = start_date + timedelta(days=day, hours=np.random.randint(0, 24))
                    # Ajouter une tendance et du bruit
                    value = base_value + (trend * day) + np.random.normal(0, base_value * 0.05)
                    
                    # Occasionnellement ajouter une valeur aberrante
                    if np.random.random() < 0.05:  # 5% de chance
                        value = value * np.random.choice([0.5, 1.5])
                    
                    data.append({
                        'Engin': engin,
                        'Paramètres Diagnostic': param,
                        'Valeur moyenne': value,
                        'Heure': date,
                        'Fonctionnel?': np.random.choice(['Oui', 'Non'], p=[0.95, 0.05])
                    })
    
    df = pd.DataFrame(data)
    
    # Filtrer sur fonctionnel par défaut
    filter_functional = st.sidebar.checkbox("Filtrer sur 'Fonctionnel? = Oui'", value=True)
    if filter_functional:
        df = df[df['Fonctionnel?'] == 'Oui']

# Vérifier si df existe et n'est pas vide
if df is None or df.empty:
    st.warning("Aucune donnée disponible. Veuillez charger un fichier Excel.")
    st.stop()

# Prétraitement des données
df_clean = df.copy()

# Détection des valeurs aberrantes
if 'Valeur moyenne' in df_clean.columns:
    df_clean = detect_outliers(df_clean, ['Engin', 'Paramètres Diagnostic'], 'Valeur moyenne')

# Filtres dans la sidebar
st.sidebar.header("Filtres")

# Sélection des engins
all_engins = sorted(df_clean['Engin'].unique())
selected_engins = st.sidebar.multiselect(
    "Sélectionnez les engins:",
    all_engins,
    default=all_engins
)

# Sélection des paramètres
all_parametres = sorted(df_clean['Paramètres Diagnostic'].unique())
selected_parametres = st.sidebar.multiselect(
    "Sélectionnez les paramètres:",
    all_parametres,
    default=all_parametres
)

# Filtre de période si 'Heure' existe
date_range = None
if 'Heure' in df_clean.columns:
    st.sidebar.subheader("Période d'analyse")
    date_min = df_clean['Heure'].min().date()
    date_max = df_clean['Heure'].max().date()
    
    date_range = st.sidebar.date_input(
        "Sélectionnez une période:",
        [date_min, date_max],
        min_value=date_min,
        max_value=date_max
    )
    
    if len(date_range) == 2:
        start_date, end_date = date_range
        df_clean = df_clean[(df_clean['Heure'].dt.date >= start_date) & 
                           (df_clean['Heure'].dt.date <= end_date)]

# Appliquer les filtres
if selected_engins:
    df_clean = df_clean[df_clean['Engin'].isin(selected_engins)]
if selected_parametres:
    df_clean = df_clean[df_clean['Paramètres Diagnostic'].isin(selected_parametres)]

# Vérifier si les données filtrées sont vides
if df_clean.empty:
    st.warning("Aucune donnée ne correspond aux filtres sélectionnés.")
    st.stop()

# Affichage des métriques clés
st.header("Statistiques générales")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Nombre d'engins", len(df_clean['Engin'].unique()))
with col2:
    st.metric("Nombre de paramètres", len(df_clean['Paramètres Diagnostic'].unique()))
with col3:
    st.metric("Nombre de mesures", len(df_clean))
with col4:
    if 'Outlier' in df_clean.columns:
        outlier_pct = (df_clean['Outlier'].sum() / len(df_clean) * 100)
        st.metric("Valeurs aberrantes", f"{outlier_pct:.1f}%")
    else:
        st.metric("Valeurs aberrantes", "N/A")

# Onglets pour différentes visualisations
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Séries temporelles", 
    "Distributions", 
    "Corrélations", 
    "Tendances normalisées",
    "Données brutes"
])

# Onglet 1: Séries temporelles
with tab1:
    st.subheader("Évolution temporelle des valeurs moyennes")
    
    # Options de visualisation
    col_options, col_download = st.columns([3, 1])
    with col_options:
        # Options pour personnaliser le graphique
        show_outliers = st.checkbox("Mettre en évidence les valeurs aberrantes", value=True)
        facet_type = st.radio(
            "Type de facettes:",
            ["Par paramètre", "Par engin"],
            horizontal=True
        )
    
    # Si 'Heure' n'existe pas, afficher un message
    if 'Heure' not in df_clean.columns:
        st.warning("La colonne 'Heure' est nécessaire pour ce graphique.")
    else:
        # Trier par date pour une meilleure visualisation
        df_clean = df_clean.sort_values('Heure')
        
        # Créer des facettes selon l'option choisie
        facet_by = 'Paramètres Diagnostic' if facet_type == "Par paramètre" else 'Engin'
        color_by = 'Engin' if facet_type == "Par paramètre" else 'Paramètres Diagnostic'
        
        # Calculer la hauteur en fonction du nombre de facettes
        num_facets = len(df_clean[facet_by].unique())
        height = max(600, 200 + 200 * num_facets)
        
        # Créer le graphique de série temporelle
        fig1 = px.line(
            df_clean,
            x='Heure',
            y='Valeur moyenne',
            color=color_by,
            facet_row=facet_by,
            facet_row_spacing=0.05,
            title=f"Évolution temporelle de la valeur moyenne par {facet_by.lower()}",
            labels={
                'Heure': 'Date & heure',
                'Valeur moyenne': 'Valeur moyenne',
                'Engin': 'Engin',
                'Paramètres Diagnostic': 'Paramètre'
            },
            line_shape='linear',
            markers=True,
            hover_data=[facet_by, color_by],
            template='plotly_white'
        )
        
        # Améliorer la mise en page
        fig1.update_layout(
            height=height,
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            margin=dict(t=100, b=50),
            hovermode='closest'
        )
        
        # Nettoyer les annotations des facettes
        fig1.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1].strip()))
        
        # Améliorer le formatage des dates sur l'axe X
        fig1.update_xaxes(
            tickformat='%d/%m/%Y\n%H:%M',
            tickangle=-45,
            tickmode='auto', 
            nticks=10
        )
        
        # Ajouter les outliers si demandé
        if show_outliers and 'Outlier' in df_clean.columns:
            outliers_df = df_clean[df_clean['Outlier']]
            if not outliers_df.empty:
                facet_values = df_clean[facet_by].unique()
                for i, facet_val in enumerate(facet_values):
                    facet_outliers = outliers_df[outliers_df[facet_by] == facet_val]
                    for color_val in facet_outliers[color_by].unique():
                        specific_outliers = facet_outliers[facet_outliers[color_by] == color_val]
                        fig1.add_trace(
                            go.Scatter(
                                x=specific_outliers['Heure'],
                                y=specific_outliers['Valeur moyenne'],
                                mode='markers',
                                marker=dict(size=10, symbol='circle-open', line=dict(width=2, color='red')),
                                name=f'Outlier - {color_val}',
                                showlegend=False,
                                hovertemplate='ANOMALIE<br>%{x}<br>%{y}<extra></extra>'
                            ),
                            row=i+1,
                            col=1
                        )
        
        st.plotly_chart(fig1, use_container_width=True)

# Onglet 2: Distributions
with tab2:
    st.subheader("Distribution des valeurs moyennes")
    
    # Options pour la visualisation
    col1, col2 = st.columns([2, 2])
    with col1:
        plot_type = st.selectbox(
            "Type de graphique:",
            ["Violin plot", "Box plot", "Histogramme"]
        )
    with col2:
        x_axis = st.selectbox(
            "Variable en axe X:",
            ["Engin", "Paramètres Diagnostic"]
        )
    
    color_var = "Paramètres Diagnostic" if x_axis == "Engin" else "Engin"
    
    if plot_type == "Violin plot":
        fig2 = px.violin(
            df_clean,
            x=x_axis,
            y='Valeur moyenne',
            color=color_var,
            box=True,
            points="outliers",
            title=f"Distribution des valeurs moyennes par {x_axis.lower()}",
            labels={'Valeur moyenne': 'Valeur moyenne'},
            template='plotly_white',
            category_orders={x_axis: sorted(df_clean[x_axis].unique())}
        )
    elif plot_type == "Box plot":
        fig2 = px.box(
            df_clean,
            x=x_axis,
            y='Valeur moyenne',
            color=color_var,
            notched=True,
            title=f"Distribution des valeurs moyennes par {x_axis.lower()}",
            labels={'Valeur moyenne': 'Valeur moyenne'},
            template='plotly_white',
            category_orders={x_axis: sorted(df_clean[x_axis].unique())}
        )
    else:  # Histogramme
        fig2 = px.histogram(
            df_clean, 
            x='Valeur moyenne',
            color=color_var,
            facet_row=x_axis,
            opacity=0.7,
            marginal="rug",
            title=f"Histogramme des valeurs moyennes par {x_axis.lower()}",
            template='plotly_white',
            barmode='overlay'
        )
        fig2.update_layout(
            height=300 + 200 * len(df_clean[x_axis].unique())
        )
        fig2.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1].strip()))
    
    fig2.update_layout(
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        margin=dict(t=100, b=50)
    )
    
    st.plotly_chart(fig2, use_container_width=True)
    
    # Afficher les statistiques descriptives
    with st.expander("Statistiques descriptives"):
        stats = df_clean.groupby(['Engin', 'Paramètres Diagnostic'])['Valeur moyenne'].describe().reset_index()
        st.dataframe(stats, use_container_width=True)
        
        # Bouton de téléchargement
        st.markdown(
            get_download_link(stats, "statistiques_descriptives.csv", "📥 Télécharger les statistiques (CSV)"),
            unsafe_allow_html=True
        )

# Onglet 3: Corrélations
with tab3:
    st.subheader("Corrélations entre paramètres")
    
    # Créer un pivot pour calculer les corrélations
    if len(selected_parametres) >= 2:
        # Pivoter le dataframe pour avoir les paramètres en colonnes
        try:
            pivot_df = df_clean.pivot_table(
                index=['Engin'] + (['Heure'] if 'Heure' in df_clean.columns else []),
                columns='Paramètres Diagnostic',
                values='Valeur moyenne',
                aggfunc='mean'
            ).reset_index()
            
            # Sélectionner uniquement les colonnes numériques (paramètres)
            corr_cols = [col for col in pivot_df.columns if col not in ['Heure', 'Engin']]
            
            # Options pour la matrice de corrélation
            col1, col2 = st.columns([2, 2])
            with col1:
                corr_method = st.selectbox(
                    "Méthode de corrélation:",
                    ["pearson", "spearman", "kendall"],
                    help="Pearson: linéaire, Spearman: monotone, Kendall: ordinal"
                )
            with col2:
                colorscale = st.selectbox(
                    "Palette de couleurs:",
                    ["RdBu_r", "Viridis", "Plasma", "Inferno", "Magma", "Cividis"]
                )
            
            # Calcul de la matrice de corrélation
            corr_matrix = pivot_df[corr_cols].corr(method=corr_method)
            
            # Créer la heatmap
            fig3 = px.imshow(
                corr_matrix,
                text_auto=True,
                aspect="auto",
                color_continuous_scale=colorscale,
                title=f"Matrice de corrélation entre les paramètres (méthode: {corr_method})",
                labels=dict(x="Paramètre", y="Paramètre", color="Corrélation"),
                template='plotly_white'
            )
            
            fig3.update_layout(
                height=600,
                width=800,
                margin=dict(t=100, l=100)
            )
            
            st.plotly_chart(fig3, use_container_width=True)
            
            # Afficher un tableau des corrélations les plus fortes
            with st.expander("Paires de paramètres avec les corrélations les plus significatives"):
                # Mettre en forme la matrice pour l'affichage
                corr_df = corr_matrix.stack().reset_index()
                corr_df.columns = ['Paramètre 1', 'Paramètre 2', 'Corrélation']
                # Filtrer pour ne pas avoir les auto-corrélations
                corr_df = corr_df[corr_df['Paramètre 1'] != corr_df['Paramètre 2']].copy()
                corr_df['Corrélation Abs'] = corr_df['Corrélation'].abs()
                # Trier par valeur absolue de corrélation
                corr_df = corr_df.sort_values('Corrélation Abs', ascending=False).drop('Corrélation Abs', axis=1)
                
                st.dataframe(corr_df, use_container_width=True)
                
                # Bouton de téléchargement
                st.markdown(
                    get_download_link(corr_df, "correlations_parametres.csv", "📥 Télécharger les corrélations (CSV)"),
                    unsafe_allow_html=True
                )
                
        except Exception as e:
            st.error(f"Erreur lors du calcul des corrélations: {e}")
            st.info("Assurez-vous d'avoir au moins 2 paramètres sélectionnés et suffisamment de données.")
    else:
        st.info("Sélectionnez au moins 2 paramètres différents pour calculer les corrélations.")

# Onglet 4: Tendances normalisées
with tab4:
    st.subheader("Tendances normalisées des paramètres")
    
    if 'Heure' not in df_clean.columns:
        st.warning("La colonne 'Heure' est nécessaire pour ce graphique.")
    else:
        # Créer une copie pour la normalisation
        df_norm = df_clean.copy()
        
        # Option pour la méthode de normalisation
        norm_method = st.radio(
            "Méthode de normalisation:",
            ["Z-score (écarts-types)", "Min-Max (0-1)"],
            horizontal=True
        )
        
        # Normalisation selon la méthode choisie
        if norm_method == "Z-score (écarts-types)":
            df_norm['Valeur normalisée'] = df_norm.groupby('Paramètres Diagnostic')['Valeur moyenne'].transform(
                lambda x: (x - x.mean()) / x.std() if x.std() != 0 else 0
            )
            y_axis_title = "Écarts-types (σ)"
        else:  # Min-Max
            df_norm['Valeur normalisée'] = df_norm.groupby('Paramètres Diagnostic')['Valeur moyenne'].transform(
                lambda x: (x - x.min()) / (x.max() - x.min()) if (x.max() - x.min()) != 0 else 0
            )
            y_axis_title = "Valeur normalisée (0-1)"
        
        # Créer un subplot pour chaque engin
        engines = sorted(df_norm['Engin'].unique())
        
        # Options pour la visualisation
        show_grid = st.checkbox("Afficher la grille", value=True)
        
        # Créer un subplot pour chaque engin
        fig4 = make_subplots(
            rows=len(engines), 
            cols=1,
            subplot_titles=[f"Tendances normalisées - {eng}" for eng in engines],
            shared_xaxes=True,
            vertical_spacing=0.1
        )
        
        # Ajouter des traces pour chaque paramètre dans chaque subplot
        for i, engine in enumerate(engines, 1):
            engine_data = df_norm[df_norm['Engin'] == engine]
            for param in sorted(engine_data['Paramètres Diagnostic'].unique()):
                param_data = engine_data[engine_data['Paramètres Diagnostic'] == param]
                fig4.add_trace(
                    go.Scatter(
                        x=param_data['Heure'],
                        y=param_data['Valeur normalisée'],
                        mode='lines+markers',
                        name=param,
                        line=dict(width=2),
                        marker=dict(size=6),
                        hovertemplate='%{y:.2f}<br>%{x}<extra>' + param + '</extra>'
                    ),
                    row=i, col=1
                )
        
        # Mise en forme
        fig4.update_layout(
            height=300 * len(engines),
            title_text=f"Évolution temporelle des paramètres normalisés par engin ({norm_method})",
            legend_title_text="Paramètre",
            showlegend=True,
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            template='plotly_white'
        )
        
        # Améliorer le formatage des axes
        fig4.update_xaxes(
            tickformat='%d/%m/%Y\n%H:%M',
            tickangle=-45,
            tickmode='auto', 
            nticks=10,
            showgrid=show_grid
        )
        
        fig4.update_yaxes(
            title_text=y_axis_title,
            showgrid=show_grid,
            zeroline=True
        )
        
        st.plotly_chart(fig4, use_container_width=True)
        
        # Explication de la normalisation
        with st.expander("À propos de la normalisation"):
            if norm_method == "Z-score (écarts-types)":
                st.markdown("""
                **Z-score (écarts-types)**
                
                - Chaque valeur est transformée en indiquant combien d'écarts-types elle s'écarte de la moyenne.
                - Une valeur de 0 correspond à la moyenne.
                - Une valeur de +1 signifie que la valeur est à 1 écart-type au-dessus de la moyenne.
                - Une valeur de -1 signifie que la valeur est à 1 écart-type en dessous de la moyenne.
                - Formule: Z = (X - moyenne) / écart-type
                
                Cette normalisation permet de comparer facilement des paramètres qui ont des échelles différentes.
                """)
            else:
                st.markdown("""
                **Normalisation Min-Max (0-1)**
                
                - Chaque valeur est transformée sur une échelle de 0 à 1.
                - Une valeur de 0 correspond au minimum observé pour ce paramètre.
                - Une valeur de 1 correspond au maximum observé pour ce paramètre.
                - Formule: X_norm = (X - min) / (max - min)
                
                Cette normalisation permet de comparer facilement des paramètres qui ont des échelles différentes.
                """)

# Onglet 5: Données brutes
with tab5:
    st.subheader("Données brutes")
    
    # Options pour l'affichage
    col1, col2, col3 = st.columns([2, 2, 2])
    with col1:
        sort_by = st.selectbox(
            "Trier par:",
            ["Engin", "Paramètres Diagnostic", "Valeur moyenne"] + 
            (["Heure"] if "Heure" in df_clean.columns else [])
        )
    with col2:
        sort_order = st.radio(
            "Ordre:",
            ["Croissant", "Décroissant"],
            horizontal=True
        )
    with col3:
        rows_per_page = st.selectbox(
            "Lignes par page:",
            [10, 25, 50, 100],
            index=1
        )
    
    # Tri des données
    ascending = sort_order == "Croissant"
    df_display = df_clean.sort_values(by=sort_by, ascending=ascending)
    
    # Pagination
    total_rows = len(df_display)
    total_pages = (total_rows // rows_per_page) + (1 if total_rows % rows_per_page > 0 else 0)
    page_number = st.number_input(
        "Page:",
        min_value=1,
        max_value=max(1, total_pages),
        value=1,
        step=1
    )
    
    start_idx = (page_number - 1) * rows_per_page
    end_idx = start_idx + rows_per_page
    df_paginated = df_display.iloc[start_idx:end_idx]
    
    # Afficher les données avec un indicateur pour les outliers
    if 'Outlier' in df_paginated.columns:
        # Fonction pour mettre en évidence les outliers
        def highlight_outliers(row):
            if row['Outlier']:
                return ['background-color: #ffcccc'] * len(row)
            return [''] * len(row)
        
        # Appliquer le style
        styled_df = df_paginated.style.apply(highlight_outliers, axis=1).format({
            'Valeur moyenne': '{:.2f}',
            'Heure': lambda x: x.strftime('%Y-%m-%d %H:%M:%S') if pd.notna(x) else ''
        })
        
        st.dataframe(styled_df, use_container_width=True, height=400)
    else:
        styled_df = df_paginated.style.format({
            'Valeur moyenne': '{:.2f}',
            'Heure': lambda x: x.strftime('%Y-%m-%d %H:%M:%S') if pd.notna(x) else ''
        })
        st.dataframe(styled_df, use_container_width=True, height=400)
    
    # Afficher les informations de pagination
    st.write(f"Affichage des lignes {start_idx + 1} à {min(end_idx, total_rows)} sur {total_rows} (Page {page_number}/{total_pages})")
    
    # Bouton de téléchargement des données brutes
    st.markdown(
        get_download_link(df_display, "donnees_brutes.csv", "📥 Télécharger toutes les données brutes (CSV)"),
        unsafe_allow_html=True
    )
    
    # Option pour afficher uniquement les outliers
    if 'Outlier' in df_display.columns:
        show_only_outliers = st.checkbox("Afficher uniquement les valeurs aberrantes", value=False)
        if show_only_outliers:
            outliers_only = df_display[df_display['Outlier']]
            if outliers_only.empty:
                st.warning("Aucune valeur aberrante trouvée dans les données filtrées.")
            else:
                # Pagination pour les outliers
                total_outlier_rows = len(outliers_only)
                total_outlier_pages = (total_outlier_rows // rows_per_page) + (1 if total_outlier_rows % rows_per_page > 0 else 0)
                outlier_page_number = st.number_input(
                    "Page (outliers):",
                    min_value=1,
                    max_value=max(1, total_outlier_pages),
                    value=1,
                    step=1
                )
                
                outlier_start_idx = (outlier_page_number - 1) * rows_per_page
                outlier_end_idx = outlier_start_idx + rows_per_page
                outliers_paginated = outliers_only.iloc[outlier_start_idx:outlier_end_idx]
                
                styled_outliers = outliers_paginated.style.apply(highlight_outliers, axis=1).format({
                    'Valeur moyenne': '{:.2f}',
                    'Heure': lambda x: x.strftime('%Y-%m-%d %H:%M:%S') if pd.notna(x) else ''
                })
                st.subheader("Valeurs aberrantes uniquement")
                st.dataframe(styled_outliers, use_container_width=True)
                st.write(f"Affichage des lignes {outlier_start_idx + 1} à {min(outlier_end_idx, total_outlier_rows)} sur {total_outlier_rows} (Page {outlier_page_number}/{total_outlier_pages})")
                st.markdown(
                    get_download_link(outliers_only, "valeurs_aberrantes.csv", "📥 Télécharger les valeurs aberrantes (CSV)"),
                    unsafe_allow_html=True
                )

# Section finale : Résumé et exportation globale
st.header("Résumé et exportation")

with st.expander("Résumé des données analysées"):
    st.write(f"""
    - **Engins analysés** : {', '.join(sorted(df_clean['Engin'].unique()))}
    - **Paramètres analysés** : {', '.join(sorted(df_clean['Paramètres Diagnostic'].unique()))}
    - **Période** : {'N/A' if 'Heure' not in df_clean.columns else f'du {df_clean["Heure"].min().strftime("%Y-%m-%d")} au {df_clean["Heure"].max().strftime("%Y-%m-%d")}'}
    - **Nombre total de mesures** : {len(df_clean)}
    - **Valeurs aberrantes détectées** : {df_clean['Outlier'].sum() if 'Outlier' in df_clean.columns else 'N/A'}
    """)

# Exportation globale
st.subheader("Exportation des visualisations")
col_export1, col_export2 = st.columns(2)
with col_export1:
    if st.button("Exporter toutes les visualisations en HTML"):
        # Créer un fichier HTML avec toutes les figures
        html_content = """
        <html>
        <head>
            <title>Visualisations des paramètres diagnostics</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        </head>
        <body>
            <h1>Visualisations des paramètres diagnostics</h1>
            <h2>Séries temporelles</h2>
            <div id="fig1"></div>
            <h2>Distributions</h2>
            <div id="fig2"></div>
            <h2>Corrélations</h2>
            <div id="fig3"></div>
            <h2>Tendances normalisées</h2>
            <div id="fig4"></div>
            <script>
        """
        if 'fig1' in locals():
            html_content += f"Plotly.newPlot('fig1', {fig1.to_json()});"
        if 'fig2' in locals():
            html_content += f"Plotly.newPlot('fig2', {fig2.to_json()});"
        if 'fig3' in locals():
            html_content += f"Plotly.newPlot('fig3', {fig3.to_json()});"
        if 'fig4' in locals():
            html_content += f"Plotly.newPlot('fig4', {fig4.to_json()});"
        html_content += """
            </script>
        </body>
        </html>
        """
        b64 = base64.b64encode(html_content.encode()).decode()
        href = f'<a href="data:text/html;base64,{b64}" download="visualisations_diagnostic.html">📥 Télécharger le rapport HTML</a>'
        st.markdown(href, unsafe_allow_html=True)
        st.success("Fichier HTML prêt à être téléchargé !")
with col_export2:
    if st.button("Exporter toutes les données filtrées"):
        st.markdown(
            get_download_link(df_clean, "donnees_filtrees_completes.csv", "📥 Télécharger toutes les données filtrées (CSV)"),
            unsafe_allow_html=True
        )
        st.success("Fichier CSV prêt à être téléchargé !")

# Pied de page
st.markdown("""
---
*Développé avec Streamlit et Plotly. Pour plus d'informations, contactez l'administrateur du système.*
""")