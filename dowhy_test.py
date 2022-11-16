
import dowhy
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import why 
import stack
from copy import deepcopy

def load_data():
    data = pd.read_csv("./dataset/data.csv")
    data.set_index("para",inplace=True)
    data = data.T
    data=data.apply(lambda x:(x-min(x))/(max(x)-min(x)))
    return data 


def main():
    G = nx.DiGraph()
    data = load_data()
    print(data.head())
    print(data.columns)
    result = pd.DataFrame(columns=data.columns, index=data.columns)
    GDD = ['emerged_GDD', 'branching_GDD', 'blooming_GDD', 'setting_pods_GDD', 'turning_yellow_GDD', 'falling_leaf_GDD']
    KDD = ['emerged_KDD', 'branching_KDD', 'blooming_KDD', 'setting_pods_KDD', 'turning_yellow_KDD', 'falling_leaf_KDD']
    EVI = ['emerged_EVI', 'branching_EVI', 'blooming_EVI', 'setting_pods_EVI', 'turning_yellow_EVI', 'falling_leaf_EVI','yield']
    ppt = ['emerged_ppt', 'branching_ppt', 'blooming_ppt', 'setting_pods_ppt', 'turning_yellow_ppt', 'falling_leaf_ppt']
    para = data.columns[1:]
    #para = ['blooming_EVI', 'setting_pods_EVI', 'turning_yellow_EVI', 'falling_leaf_EVI', 'yield']
    G.add_node('emerged_GDD')
    # nx.write_gml(G, './dataset/model.gml')
    S = stack.stack_list()
    S.push('emerged_GDD')

    for i in para:
        T = deepcopy(S)
        G.add_node(i)
        while not T.is_Empty():
            node = T.top()
            T.pop()
            if node in EVI and i not in EVI:
                continue
            if node in ppt and i not in EVI:
                continue
            if node in GDD and i in KDD:
                continue 
            if node in KDD and i in GDD:
                continue 
            G.add_edge(node,i)
            father = list(G.predecessors(node))
            for fa in father:
                G.add_edge(fa,i)
            causal_graph = '\n'.join(nx.generate_gml(G))
            model= dowhy.CausalModel(data = data,
                                    graph = causal_graph,
                                    treatment = node,
                                    outcome = i
                                    )
            model.view_model()
            ate = why.backdoor_ate(model)
            nde = why.mediation_nde(model)
            if nde == None:
                if abs(ate) < 0.10:
                    G.remove_edge(node,i)
                else:
                    G[node][i]['weight'] = round(ate,2)
                    result[node][i] = round(ate,2)
            else:
                if abs(nde) <0.10:
                    G.remove_edge(node,i)
                else:
                    G[node][i]['weight'] = round(nde,2)
                    result[node][i] = round(ate,2)
            print('ate = {}'.format(ate))
            print('nde = {}'.format(nde))
            for fa in father:
                G.remove_edge(fa,i)

        S.push(i)

    # #画图
    pos = nx.spring_layout(G)
    labels = nx.get_edge_attributes(G,'weight')
    fig=plt.figure()
    nx.draw(G, pos, with_labels=True, font_weight='bold')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    plt.show()
    print(result)
    result.to_csv('result.csv')

if __name__ == '__main__':
    main()