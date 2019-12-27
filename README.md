# dsl-project

To install dependencies:

    $ pip install -r requirements.txt

To run:

    python run.py
              
              {
                hierarchical_attention_network,
                pruned_hierarchical_attention_network,
                hierarchical_sparsemax_attention_network,
                mlp
              }

              [-dataset DATASET]
              [-resources RESOURCES] 
              
              [-layers LAYERS] 
              [-hidden_sizes HIDDEN_SIZES] 
              [-activation ACTIVATION] 
              [-dropout DROPOUT] 
              
              [-epochs EPOCHS] 
              [-optimizer {sgd,adam}] 
              [-learning_rate LEARNING_RATE] 
              [-l2_decay L2_DECAY] 
              [-batch_size BATCH_SIZE] 
              
              [-cuda] 
              [-quiet] 
              [-tqdm] 
              [-save_plot]
              
