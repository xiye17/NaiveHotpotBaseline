
def add_common_args(parser):
    parser.add_argument('--cache_dir', default='hf_cache', type=str, help='custom cache dir')

    return parser
