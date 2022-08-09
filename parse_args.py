import argparse

parser = argparse.ArgumentParser()

# ---------------------------File--------------------------- #
parser.add_argument('--data_path',          default='../data')
parser.add_argument('--mobility_adj',       default='/mobility_adj.npy')
parser.add_argument('--poi_similarity',     default='/poi_similarity.npy')
parser.add_argument('--source_adj',         default='/source_adj.npy')
parser.add_argument('--destination_adj',    default='/destination_adj.npy')
parser.add_argument('--mh_cd',              default='/mh_cd.json')
parser.add_argument('--crime_counts',       default='/crime_counts.npy')
parser.add_argument('--neighbor',           default='/neighbor.npy')

# ---------------------------Data--------------------------- #
parser.add_argument('--regions_num',    type=int,    default=180)
parser.add_argument('--importance_k',   type=int,    default=10)

# ---------------------------Model--------------------------- #
parser.add_argument('--device',                      default='cuda')
parser.add_argument('--embedding_size', type=int,    default=144)  # 144
parser.add_argument('--learning_rate',  type=float,  default=0.001)  # 0.001
parser.add_argument('--epochs',         type=int,    default=2000)  # 2000
parser.add_argument('--dropout',        type=float,  default=0.1)
parser.add_argument('--gcn_layers',     type=int,    default=3)  # 4

# ---------------------------Save--------------------------- #
parser.add_argument('--save_folder', default='./save_folder')

args = parser.parse_args()

