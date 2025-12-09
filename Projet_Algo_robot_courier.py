# ============================================================================
# PROJET ALGORITHMIQUE - Robo-Courier
# Version Finale
#
# Auteurs : Raphaël Maul et Alexandre Raffin
# ============================================================================

import matplotlib
matplotlib.use('TkAgg') # Force l'utilisation du backend TkAgg pour l'interface graphique (fenêtres interactives)
import networkx as nx  # Pour la gestion des graphes
import matplotlib.pyplot as plt  # Pour le tracé graphique
import matplotlib.animation as animation  # Pour l'animation du robot
from matplotlib.widgets import Button  # Pour le bouton Pause
import heapq  # Pour la file de priorité (utilisée dans Dijkstra/A*)
import math   # Pour les calculs de distances (racine carrée)
import random # Pour la génération de nombres aléatoires (risques)
import time   # Pour mesurer le temps d'exécution

# ----------------------------------------------------------------------------
# SECTION 1 : Paramètres généraux
# ----------------------------------------------------------------------------

FRAMES_PER_EDGE = 5    # Nombre d'images pour dessiner le passage d'une arête (fluidité)
INTERVAL_MS = 20     # Intervalle en millisecondes entre chaque image (vitesse)
SCALE_FACTOR = 4       # Facteur d'agrandissement des coordonnées pour l'affichage
NODE_SIZE = 600        # Taille visuelle des nœuds (cercles) sur le graphe
FONT_SIZE = 8          # Taille de la police pour les noms des lieux
LABEL_OFFSET = 0.25    # Décalage des étiquettes de poids sur les arêtes

# ----------------------------------------------------------------------------
# SECTION 2 : Données de la carte
# ----------------------------------------------------------------------------

# Dictionnaire des positions (x, y) de chaque lieu (nœud)
pos_base = {
    'Combes': (2, 6), 'Depot': (4, 4.5), 'Mcdo': (6, 3), 'Condorcet': (4, 3),
    'Alex': (4, 1), 'Durque': (3, 2), 'HG': (1, 2), 'Raph': (0, -1.5),
    'Parc': (2, 1), 'Mairie': (4.5, -1), 'SaintP': (6, -1), 'Grimoire': (7, 2), 'Shao': (5, 2),
    'One Club': (13, -3), 'Leclerc': (14.5, 3), 'BK': (14.25, -2.5),
    'Colisée': (15, -2), 'Statue': (11, 2), 'Gare': (13.5, -1.5),
    'Basilique': (13, 10), 'Château': (15, 8.5), 'Camping': (15.5, 11.5),
    'Vignes': (12, 11.5), 'Hotel-Dieu': (14.25, 9.5), 'Belenium': (14.5, 11),
    'Cascade': (2, 10), 'Pyramide': (1, 10), 'Cathedrale': (0, 11),
    'Théâtre': (2.5, 11.5), 'Inter': (-1, 14), 'Temple': (1, 13), 'Bistrot': (4, 11),
}

# Création des positions mises à l'échelle pour un affichage plus lisible
pos_esthetique = {k: (v[0]*SCALE_FACTOR, v[1]*SCALE_FACTOR) for k,v in pos_base.items()}

# Définition des voisins pour chaque lieu (liste d'adjacence)
graph_links = {
    'Combes': ['Depot', 'Durque', 'HG', 'Cascade', 'Cathedrale'],
    'Depot': ['Combes', 'Mcdo', 'Durque'],
    'Mcdo': ['Depot', 'Condorcet'],
    'Condorcet': ['Mcdo', 'Alex'],
    'Alex': ['Condorcet', 'Durque', 'SaintP', 'Shao'],
    'Durque': ['Combes', 'Depot', 'Alex', 'HG', 'Parc', 'Mairie'],
    'HG': ['Combes', 'Durque', 'Raph'],
    'Raph': ['HG', 'Parc'],
    'Parc': ['Durque', 'Raph'],
    'Mairie': ['Durque', 'SaintP'],
    'SaintP': ['Alex', 'Mairie', 'Grimoire'],
    'Grimoire': ['SaintP', 'Shao', 'One Club', 'Statue'],
    'Shao': ['Alex', 'Grimoire'],
    'One Club': ['Grimoire', 'Gare', 'BK'],
    'Leclerc': ['Statue', 'Colisée', 'Hotel-Dieu', 'Château'],
    'BK': ['One Club', 'Gare', 'Colisée'],
    'Colisée': ['Leclerc', 'BK'],
    'Statue': ['Leclerc', 'Grimoire', 'Gare'],
    'Gare': ['One Club', 'BK', 'Statue'],
    'Basilique': ['Belenium', 'Hotel-Dieu', 'Vignes'],
    'Château': ['Leclerc', 'Camping'],
    'Camping': ['Château', 'Belenium'],
    'Vignes': ['Basilique', 'Bistrot'],
    'Hotel-Dieu': ['Leclerc', 'Basilique'],
    'Belenium': ['Camping', 'Basilique'],
    'Cascade': ['Combes', 'Pyramide', 'Bistrot'],
    'Pyramide': ['Cascade', 'Cathedrale'],
    'Cathedrale': ['Combes', 'Pyramide', 'Temple'],
    'Théâtre': ['Temple', 'Bistrot'],
    'Inter': ['Temple'],
    'Temple': ['Inter', 'Cathedrale', 'Théâtre'],
    'Bistrot': ['Cascade', 'Théâtre', 'Vignes'],
}

# ----------------------------------------------------------------------------
# SECTION 3 : Définition des heuristiques
# ----------------------------------------------------------------------------

def get_euclidean_distance(n1,n2,pos_map):
 # Calcule la distance 'à vol d'oiseau' (Pythagore) entre deux nœuds n1 et n2 
    x1,y1 = pos_map[n1]; x2,y2 = pos_map[n2]
    return math.sqrt((x1-x2)**2 + (y1-y2)**2)

def heuristic_euclidean(node,goal,pos):
    # Heuristique pour A* basée sur la distance Euclidienne 
    return get_euclidean_distance(node,goal,pos)

def heuristic_manhattan(node,goal,pos):
    # Heuristique pour A* basée sur la distance de Manhattan (|dx| + |dy|) 
    x1,y1 = pos[node]; x2,y2 = pos[goal]
    return abs(x1-x2)+abs(y1-y2)

def generate_map_data(links,positions): 
    
    # Calcule le poids de chaque arête en fonction de la distance réelle.
    map_data = {}
    for node,neighbors in links.items():
        if node not in positions: continue
        # Création des arêtes pondérées
        weighted = [(n,get_euclidean_distance(node,n,positions)) for n in neighbors if n in positions]
        map_data[node]=weighted
    return map_data

# ----------------------------------------------------------------------------
# SECTION 4 : Algorithmes Dijkstra et A*
# ----------------------------------------------------------------------------
# Algorithme de Dijkstra
def dijkstra_path(map_data,start,goal,risks_snapshot=None):
    # Initialisation des distances à l'infini pour tous les nœuds
    distances = {n:float('inf') for n in map_data}
    distances[start]=0 # La distance au point de départ est 0
    parents={start:None} # Pour reconstruire le chemin
    
    # File de priorité contenant (coût_actuel, nœud)
    pq=[(0,start)]
    
    explored_edges=[] # Liste pour l'animation des arêtes explorées
    nodes_visited=0   # Compteur de nœuds traités
    modified_weights={} # Pour stocker les arêtes impactées par des risques
    
    while pq:
        nodes_visited+=1
        # On récupère le nœud avec la plus petite distance connue
        cur_dist,cur=heapq.heappop(pq)
        
        # Si on atteint la destination, on arrête
        if cur==goal: break
        
        # Si on a trouvé un chemin plus long que celui déjà connu, on ignore
        if cur_dist>distances[cur]: continue
        
        # Exploration des voisins
        for n,w in map_data[cur]:
            edge_key = tuple(sorted((cur,n)))
            
            # Application de la pénalité si l'arête est à risque (dans le snapshot)
            if risks_snapshot and risks_snapshot.get(edge_key, False):
                w += 500 # Ajout d'un coût très élevé pour simuler un blocage
                modified_weights[edge_key] = w
            
            # Enregistrement pour l'animation (arête explorée)
            if (n,cur) not in explored_edges: explored_edges.append((cur,n))
            
            # Relâchement de l'arête (Relaxation)
            new_dist = cur_dist + w
            if new_dist < distances[n]:
                distances[n] = new_dist
                parents[n] = cur
                heapq.heappush(pq,(new_dist,n))
    
    # Reconstruction du chemin en remontant les parents
    path=[]
    c=goal
    while c:
        path.append(c); c=parents.get(c)
    path.reverse() # On remet le chemin dans l'ordre Départ -> Arrivée
    
    # Calcul du coût réel du chemin trouvé (incluant pénalités)
    path_cost = distances[goal] if path and path[0]==start else 0
    
    # Gestion du cas où aucun chemin n'est trouvé
    if not path or path[0]!=start: return [],explored_edges,nodes_visited,modified_weights, 0
    return path,explored_edges,nodes_visited,modified_weights, path_cost

# Algorithme A*
def a_star_path(map_data,pos,start,goal,heuristic_func=heuristic_euclidean,risks_snapshot=None):
    # Initialisation des scores
    g_score={n:float('inf') for n in map_data}; g_score[start]=0 # Coût réel depuis le départ
    # f_score = g_score + heuristique (estimation jusqu'à l'arrivée)
    f_score={n:float('inf') for n in map_data}; f_score[start]=heuristic_func(start,goal,pos)
    
    open_set=[(f_score[start],start)] # File de priorité basée sur f_score
    came_from={start:None} # Pour reconstruire le chemin
    
    edges=[]; nodes_visited=0; modified_weights={}
    
    while open_set:
        nodes_visited+=1
        _,cur=heapq.heappop(open_set) # On prend le nœud avec le f_score le plus bas
        
        if cur==goal: break
        
        for n,w in map_data[cur]:
            edge_key = tuple(sorted((cur,n)))
            # Vérification des risques (pénalités)
            if risks_snapshot and risks_snapshot.get(edge_key, False):
                w += 500
                modified_weights[edge_key] = w
            
            tent_g = g_score[cur] + w # Coût g temporaire
            
            if (n,cur) not in edges: edges.append((cur,n))
            
            # Si ce chemin est meilleur que le précédent
            if tent_g < g_score[n]:
                came_from[n] = cur
                g_score[n] = tent_g
                # f = g + h
                f_score[n] = tent_g + heuristic_func(n,goal,pos)
                heapq.heappush(open_set,(f_score[n],n))
                
    # Reconstruction du chemin
    path=[]; c=goal
    while c:
        path.append(c); c=came_from.get(c)
    path.reverse()
    
    path_cost = g_score[goal] if path and path[0]==start else 0

    if not path or path[0]!=start: return [],edges,nodes_visited,modified_weights, 0
    return path,edges,nodes_visited,modified_weights, path_cost


# ----------------------------------------------------------------------------
# SECTION 5 : Fonctions Optionnelles 
# ----------------------------------------------------------------------------
# Gestion de la capacité du robot 
def solve_vrp_capacity(map_data,pos,start,goals,capacity,algo='dijkstra',heu_func=heuristic_euclidean, enable_risks=True, risks_snapshot=None, rng_seed=None):
    """
    Fonction principale de gestion de la tournée.
    Gère la capacité du robot et les retours au dépôt.
    """
    path_nodes=[];all_edges=[];edge_colors=[];all_mods={};nodes_visited_total=0
    # Palette de couleurs pour distinguer les différents voyages
    ROUND_COLORS=['#FF0000','#0000FF','#00AA00','#800080','#FFA500','#A52A2A']
    round_idx=0; cur=start; pending=goals.copy(); path_nodes.append(start)
    
    total_cost_acc = 0
    full_delivery_order = []

    # Génération ou récupération des risques (grèves/bouchons)
    if risks_snapshot is None and enable_risks:
        rng = random.Random(rng_seed) if rng_seed is not None else random.Random()
        risks_snapshot = generate_risks_snapshot_from_rng(rng)
    elif not enable_risks:
        risks_snapshot = {}

    # Boucle tant qu'il reste des colis à livrer
    while pending:
        color=ROUND_COLORS[round_idx%len(ROUND_COLORS)]; round_idx+=1
        
        # Chargement du robot selon sa capacité (0 = infinie)
        load=pending[:capacity] if capacity>0 else pending.copy()
        for pkg in load: pending.remove(pkg)
        
        # Calcul de l'ordre de livraison optimal (Glouton) pour ce chargement
        order=get_nearest_neighbor_path(map_data,pos,cur,load)
        full_delivery_order.extend(order)

        # Exécution des trajets pour chaque point de ce chargement
        for tgt in order:
            if algo=='dijkstra':
                seg,expl,nv,mods,cost = dijkstra_path(map_data,cur,tgt,risks_snapshot=risks_snapshot)
            else:
                seg,expl,nv,mods,cost = a_star_path(map_data,pos,cur,tgt,heuristic_func=heu_func,risks_snapshot=risks_snapshot)
            
            if seg:
                path_nodes.extend(seg[1:]); edge_colors.extend([color]*len(seg[1:]))
                total_cost_acc += cost
            all_edges.extend(expl); nodes_visited_total+=nv; all_mods.update(mods)
            cur=tgt
            
        # Logique de retour au dépôt :
        # - Si on a encore des colis en attente et une capacité limitée
        # - Ou si on a fini mais qu'on n'est pas au dépôt
        must_return = (capacity > 0 and pending) or (capacity == 0 and cur != start) or (capacity > 0 and not pending and cur != start)
        
        if must_return:
            if algo=='dijkstra': seg,expl,nv,mods,cost = dijkstra_path(map_data,cur,start,risks_snapshot=risks_snapshot)
            else: seg,expl,nv,mods,cost = a_star_path(map_data,pos,cur,start,heuristic_func=heu_func,risks_snapshot=risks_snapshot)
            if seg:
                path_nodes.extend(seg[1:]); edge_colors.extend([color]*len(seg[1:]))
                total_cost_acc += cost
            all_edges.extend(expl); nodes_visited_total+=nv; all_mods.update(mods)
            cur=start

    return path_nodes, all_edges, nodes_visited_total, edge_colors, all_mods, total_cost_acc, full_delivery_order

# Gestion des risques (manifestations/grèves)
# Définition des arêtes avec une probabilité d'incident
RISKY_EDGES = {
    ('Leclerc','Château'):0.05,
    ('HG','Durque'):0.90,
    ('Bistrot','Vignes'):0.33,
    ('SaintP','Grimoire'):0.14,    
}

def generate_risks_snapshot_from_rng(rng: random.Random):
    """
    Génère un état figé des risques pour toute la simulation.
    Pour chaque arête à risque, tire au sort si elle est bloquée.
    """
    snap = {}
    for (u,v),prob in RISKY_EDGES.items():
        key = tuple(sorted((u,v)))
        snap[key] = rng.random() < prob
    return snap

# ----------------------------------------------------------------------------
# SECTION 6 : Méthode pour ordonner les livraisons (Glouton)
# ----------------------------------------------------------------------------

def get_nearest_neighbor_path(map_data,pos,start,goals):
    """
    Implémente une stratégie Gloutonne (Nearest Neighbor)
    Choisit toujours la livraison la plus proche de la position actuelle.
    """
    order=[];cur=start; pending=goals.copy()
    while pending:
        nearest=None; min_dist=float('inf')
        for g in pending:
            d=get_euclidean_distance(cur,g,pos)
            if d<min_dist: min_dist=d; nearest=g
        order.append(nearest); pending.remove(nearest); cur=nearest
    return order

# ----------------------------------------------------------------------------
# SECTION 7 : Animation et affichage
# ----------------------------------------------------------------------------

def get_edge_label_position(pos,u,v,offset=0.1):
    """ Calcule la position pour afficher le texte (poids) à côté de l'arête """
    x1,y1=pos[u]; x2,y2=pos[v]
    mid_x=(x1+x2)/2; mid_y=(y1+y2)/2
    # Calcul d'un vecteur perpendiculaire pour décaler le texte
    dx=x2-x1; dy=y2-y1; perp_dx=-dy; perp_dy=dx
    length=(perp_dx**2+perp_dy**2)**0.5
    if length>0: perp_dx/=length; perp_dy/=length
    return (mid_x+perp_dx*offset, mid_y+perp_dy*offset)

def _draw_static_base(ax,title,G,pos_to_draw,blue_edges,edge_labels,modified_weights=None):
    """ Dessine le fond de carte statique (nœuds, arêtes grises, labels) """
    if modified_weights is None: modified_weights={}
    ax.clear(); ax.set_title(title)
    # Dessin des nœuds
    nx.draw_networkx_nodes(G,pos_to_draw,ax=ax,node_color='lightgrey',node_size=NODE_SIZE)
    # Dessin des arêtes de base
    nx.draw_networkx_edges(G,pos_to_draw,ax=ax,edge_color='#cccccc',alpha=0.5)
    # Dessin des noms des lieux
    nx.draw_networkx_labels(G,pos_to_draw,ax=ax,font_size=FONT_SIZE)
    # Dessin des arêtes explorées par l'algo (en bleu clair)
    if len(blue_edges)<500: nx.draw_networkx_edges(G,pos_to_draw,ax=ax,edgelist=blue_edges,edge_color='blue',width=2,alpha=0.1)
    
    # Gestion de l'affichage des poids (rouges si modifiés, verts sinon)
    labels_to_draw={}
    for (u,v),w in edge_labels.items():
        edge_key=tuple(sorted((u,v)))
        if edge_key in modified_weights: labels_to_draw[edge_key]=(f"{modified_weights[edge_key]:.1f}", 'red', 'yellow')
        else: labels_to_draw[edge_key]=(f"{w:.1f}", 'darkgreen','white')
    
    # Affichage manuel des étiquettes de poids
    drawn=set()
    for (u,v),(txt,txt_col,bg_col) in labels_to_draw.items():
        edge_key=tuple(sorted((u,v)))
        u_k, v_k = u, v
        if edge_key not in drawn and u_k in pos_to_draw and v_k in pos_to_draw:
            label_pos=get_edge_label_position(pos_to_draw,u_k,v_k,LABEL_OFFSET)
            edge_col='red' if txt_col=='red' else 'none'
            font_w='bold' if txt_col=='red' else 'normal'
            ax.text(label_pos[0],label_pos[1],txt,fontsize=7,fontweight=font_w,color=txt_col,ha='center',va='center',
                    bbox=dict(boxstyle="round,pad=0.2",facecolor=bg_col,edgecolor=edge_col,alpha=0.8))
            drawn.add(edge_key)

def _draw_animated_path(ax,G,pos_to_draw,path,path_edges,current_edge_index,percentage,custom_colors=None):
    """ Dessine le chemin parcouru et le robot en mouvement """
    # Dessiner les segments terminés
    if current_edge_index>0:
        for i in range(current_edge_index):
            u,v=path_edges[i]
            c=custom_colors[i] if custom_colors and i<len(custom_colors) else 'red'
            nx.draw_networkx_edges(G,pos_to_draw,ax=ax,edgelist=[(u,v)],edge_color=c,width=4)
    
    # Dessiner le segment en cours (interpolation linéaire pour le mouvement)
    xt, yt = 0, 0 # Coords du robot
    if current_edge_index<len(path_edges):
        u,v=path_edges[current_edge_index]
        x0,y0=pos_to_draw[u]; x1,y1=pos_to_draw[v]
        # Calcul de la position intermédiaire
        xt=x0+(x1-x0)*percentage; yt=y0+(y1-y0)*percentage
        curr_c=custom_colors[current_edge_index] if custom_colors and current_edge_index<len(custom_colors) else 'red'
        ax.plot([x0,xt],[y0,yt],color=curr_c,linewidth=4)
        ax.set_title(f"{ax.get_title().splitlines()[0]}\nNavigation : {u} → {v}")
        
        # --- MODIF: DESSINER LE ROBOT (Rond Noir) ---
        ax.plot(xt, yt, 'o', color='black', markersize=8, zorder=10)

    # Dessiner les noeuds déjà visités
    nodes_draw=path[:current_edge_index+1]
    if percentage>=1.0 and (current_edge_index+1)<len(path):
        nodes_draw.append(path[current_edge_index+1])
    
    # Gestion des couleurs des nœuds
    node_colors_list=[]
    if custom_colors:
        if len(custom_colors)>0: node_colors_list.append(custom_colors[0])
        else: node_colors_list.append('red')
        for i in range(1,len(nodes_draw)):
            idx=i-1
            if idx<len(custom_colors): node_colors_list.append(custom_colors[idx])
            else: node_colors_list.append(custom_colors[-1])
    else: node_colors_list=['#ff8888']*len(nodes_draw)
    
    nx.draw_networkx_nodes(G,pos_to_draw,ax=ax,nodelist=nodes_draw,node_color=node_colors_list,node_size=NODE_SIZE)

def get_visible_mods(all_mods,visited_nodes):
    """ Filtre les modificateurs de poids pour n'afficher que ceux pertinents pour les nœuds visités """
    vis={}; visited_set=set(visited_nodes)
    for (u,v),w in all_mods.items():
        if u in visited_set or v in visited_set: vis[(u,v)]=w
    return vis

def on_escape_key(event):
    """ Callback pour fermer la fenêtre avec la touche Echap """
    if event.key == 'escape':
        plt.close('all')
        print("\n[INFO] Fermeture de l'application via Echap.")

def animate_single_graph(title,G,pos_to_draw,path,explored_edges,edge_labels,custom_edge_colors=None,modified_weights=None):
    """ Fonction principale d'animation pour un seul graphe """
    if modified_weights is None: modified_weights={}
    path_edges=list(zip(path,path[1:]))
    path_set=set(path_edges)|set([(v,u) for u,v in path_edges])
    # Identification des arêtes explorées mais non empruntées (en bleu)
    blue_edges=[e for e in explored_edges if e not in path_set and (e[1],e[0]) not in path_set]
    
    fig,ax=plt.subplots(figsize=(14,10)); plt.subplots_adjust(left=0.02,right=0.98,top=0.95,bottom=0.1)
    
    # Ajout du listener pour Echap
    fig.canvas.mpl_connect('key_press_event', on_escape_key)
    
    total_frames=len(path_edges)*FRAMES_PER_EDGE
    
    # Fonction de mise à jour appelée à chaque frame
    def update_single(frame):
        idx=frame//FRAMES_PER_EDGE; pct=(frame%FRAMES_PER_EDGE+1)/FRAMES_PER_EDGE
        nodes_visited_so_far=path[:idx+1]
        visible_mods=get_visible_mods(modified_weights,nodes_visited_so_far)
        _draw_static_base(ax,title,G,pos_to_draw,blue_edges,edge_labels,visible_mods)
        _draw_animated_path(ax,G,pos_to_draw,path,path_edges,idx,pct,custom_edge_colors)
        
    ani=animation.FuncAnimation(fig,update_single,frames=total_frames,interval=INTERVAL_MS,repeat=False)
    
    # Ajout du bouton Pause
    ax_button=plt.axes([0.81,0.01,0.1,0.05])
    btn=Button(ax_button,'Pause',color='lightgoldenrodyellow',hovercolor='0.975')
    state={'paused':False}
    def toggle_pause(event):
        if state['paused']: ani.event_source.start(); btn.label.set_text('Pause'); state['paused']=False
        else: ani.event_source.stop(); btn.label.set_text('Reprendre'); state['paused']=True
    btn.on_clicked(toggle_pause); plt.show()

def animate_full_comparison(G,pos_to_draw,edge_labels,d_res,a_res,algo_name_2="A*"):
    """ Fonction d'animation double : compare Dijkstra (gauche) et A* (droite) """
    d_path,d_edges,d_count,d_colors,d_mods,d_cost,d_order=d_res
    a_path,a_edges,a_count,a_colors,a_mods,a_cost,a_order=a_res
    
    # Préparation des arêtes bleues (explorées) pour Dijkstra
    d_edges_set=set(list(zip(d_path,d_path[1:])))|set([(v,u) for u,v in zip(d_path,d_path[1:])])
    d_blue=[e for e in d_edges if e not in d_edges_set and (e[1],e[0]) not in d_edges_set]
    # Préparation des arêtes bleues (explorées) pour A*
    a_edges_set=set(list(zip(a_path,a_path[1:])))|set([(v,u) for u,v in zip(a_path,a_path[1:])])
    a_blue=[e for e in a_edges if e not in a_edges_set and (e[1],e[0]) not in a_edges_set]
    
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(18,10)); plt.subplots_adjust(left=0.02,right=0.98,top=0.90,bottom=0.1,wspace=0.1)
    fig.suptitle(f"Comparaison : Dijkstra vs {algo_name_2}",fontsize=16)
    fig.canvas.mpl_connect('key_press_event', on_escape_key)

    max_edges=max(len(d_path),len(a_path)); total_frames=max_edges*FRAMES_PER_EDGE
    
    def update(frame):
        idx=frame//FRAMES_PER_EDGE; pct=(frame%FRAMES_PER_EDGE+1)/FRAMES_PER_EDGE
        
        # Dijkstra (Mise à jour panneau gauche)
        if idx < len(d_path)-1:
            d_visited=d_path[:idx+1]
            d_vis=get_visible_mods(d_mods,d_visited)
            t1=f"Dijkstra\nExplorés: {d_count} | Coût: {d_cost:.1f}"
            _draw_static_base(ax1,t1,G,pos_to_draw,d_blue,edge_labels,d_vis)
            _draw_animated_path(ax1,G,pos_to_draw,d_path,list(zip(d_path,d_path[1:])),idx,pct,d_colors)
        
        # A* (Mise à jour panneau droit)
        if idx < len(a_path)-1:
            a_visited=a_path[:idx+1]
            a_vis=get_visible_mods(a_mods,a_visited)
            t2=f"{algo_name_2}\nExplorés: {a_count} | Coût: {a_cost:.1f}"
            _draw_static_base(ax2,t2,G,pos_to_draw,a_blue,edge_labels,a_vis)
            _draw_animated_path(ax2,G,pos_to_draw,a_path,list(zip(a_path,a_path[1:])),idx,pct,a_colors)

    ani=animation.FuncAnimation(fig,update,frames=total_frames,interval=INTERVAL_MS,repeat=False)
    ax_button=plt.axes([0.81,0.01,0.1,0.05]); btn=Button(ax_button,'Pause',color='lightgoldenrodyellow',hovercolor='0.975')
    state={'paused':False}
    def toggle_pause(event):
        if state['paused']: ani.event_source.start(); btn.label.set_text('Pause'); state['paused']=False
        else: ani.event_source.stop(); btn.label.set_text('Reprendre'); state['paused']=True
    btn.on_clicked(toggle_pause); plt.show()

# ----------------------------------------------------------------------------
# SECTION 8 : Affichage des résultats
# ----------------------------------------------------------------------------

def print_stats(algo_name, res, execution_time):
    """ Affiche un tableau récapitulatif dans la console """
    path_nodes, all_edges, nodes_visited, edge_colors, all_mods, total_cost, delivery_order = res
    print(f"\n--- RÉSULTATS {algo_name.upper()} ---")
    print(f" > Temps de calcul       : {execution_time:.6f} secondes")
    print(f" > Longueur du trajet    : {total_cost:.2f} (unités de coût)")
    print(f" > Nœuds explorés (algo) : {len(all_edges)}")
    print(f" > Nœuds visités (path)  : {len(path_nodes)}")
    print(f" > Ordre de livraison    : {' -> '.join(delivery_order)}")
    print(f" > Chemin complet        : {' -> '.join(path_nodes)}")

# ----------------------------------------------------------------------------
# SECTION 9 : Programme principal
# ----------------------------------------------------------------------------

def main():
    print("\n=== PROJET ROBO-COURIER) ===")
    
    # Génération des données du graphe
    map_data=generate_map_data(graph_links,pos_esthetique)
    G=nx.Graph()
    for node,neighbors in map_data.items():
        for n,w in neighbors: G.add_edge(node,n,weight=w)
    edge_labels={(u,v):d['weight'] for u,v,d in G.edges(data=True)}

    start="Depot"
    print(f"Départ fixe : {start}")
    print(f"Lieux disponibles : {', '.join(sorted(map_data.keys()))}\n")

    # 1. Choix des lieux par l'utilisateur
    goals_input=input("1. Donnez les lieux de livraison (ex: Mairie, Mcdo, Gare) : ")
    goals=[g.strip() for g in goals_input.split(',') if g.strip() in map_data]
    if not goals: print("Erreur : Aucune destination valide."); return

    # 2. Options modulaires (Extensions)
    print("\n--- OPTIONS & EXTENSIONS ---")
    
    # Option Capacité (VRP)
    use_capacity = input("  > Voulez-vous activer l'option de capacité limitée ? (o/n) : ").strip().lower() == 'o'
    capacity = 0
    if use_capacity:
        try: capacity=int(input("     > Donnez la capacité du robot : "))
        except: capacity=0
    else:
        print("     > Capacité illimitée.")

    # Option Manifestations (Risques)
    use_risks = input("  > Y a-t-il des étudiants en colère, des gilets jaunes et des alcooliques sur les routes aujourd'hui ? (o/n) : ").strip().lower() == 'o'
    if not use_risks:
        print("     > Le trafic est fluide aujourd'hui")
    else:
        print("     > Attention aux zones à risque !")

    # 3. Choix Algorithme
    print("\n--- ALGORITHME ---")
    algo_choice=input("  > Choisissez l'algorithme (dijkstra / a_star / both) : ").strip().lower()
    heu_func=heuristic_euclidean; heu_name="A*"
    if algo_choice in ['a_star','both']:
        h_str=input("     > Coisissez l'heuristique ? (1=Euclidienne, 2=Manhattan) : ").strip()
        if h_str=='2': heu_func=heuristic_manhattan; heu_name="A* (Manhattan)"
        else: heu_name="A* (Euclidienne)"

    # Graine partagée pour assurer que les deux algorithmes (en mode 'both') affrontent exactement les mêmes aléas (manifestations).
    shared_seed = random.randint(0, 2**30 - 1)
    
    # --- EXÉCUTION ---
    
    if algo_choice=='both':
        # Mode comparaison
        rng_for_shared = random.Random(shared_seed)
        shared_snapshot = generate_risks_snapshot_from_rng(rng_for_shared) if use_risks else {}

        # Exécution de Dijkstra
        t0 = time.perf_counter()
        d_res = solve_vrp_capacity(map_data,pos_esthetique,start,goals,capacity,algo='dijkstra',heu_func=heu_func, enable_risks=use_risks, risks_snapshot=shared_snapshot)
        t1 = time.perf_counter()
        print_stats("Dijkstra", d_res, t1-t0)

        # Exécution de A*
        t0 = time.perf_counter()
        a_res = solve_vrp_capacity(map_data,pos_esthetique,start,goals,capacity,algo='a_star',heu_func=heu_func, enable_risks=use_risks, risks_snapshot=shared_snapshot)
        t1 = time.perf_counter()
        print_stats(heu_name, a_res, t1-t0)
        
        # Lancement de l'animation double
        animate_full_comparison(G,pos_esthetique,edge_labels,d_res,a_res,algo_name_2=heu_name)
    
    else:
        # Mode simple (un seul algo)
        t0 = time.perf_counter()
        res = solve_vrp_capacity(map_data,pos_esthetique,start,goals,capacity,algo=algo_choice,heu_func=heu_func, enable_risks=use_risks, rng_seed=shared_seed)
        t1 = time.perf_counter()
        
        print_stats(algo_choice if algo_choice!='a_star' else heu_name, res, t1-t0)
        
        full_path,explored,visited,colors,mods,cost,order = res
        title_disp=f"Tournée {heu_name if algo_choice=='a_star' else 'Dijkstra'} (Cap: {capacity}) - Coût: {cost:.1f}"
        # Lancement de l'animation simple
        animate_single_graph(title_disp,G,pos_esthetique,full_path,explored,edge_labels,custom_edge_colors=colors,modified_weights=mods)

if __name__=="__main__":
    main()