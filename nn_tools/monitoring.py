# ==============================================================================
# ================================ Tensorboard =================================
# ==============================================================================

# # Install ngrok
# !wget https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip
# !unzip ngrok-stable-linux-amd64.zip


# # Specify tensorboard log directory
# TB_LOG_DIR = './tb_log' # tensorboard log directory

# # run tensorboard
# get_ipython().system_raw(
#     'tensorboard --logdir {} --host 0.0.0.0 --port 6006 &'
#     .format(TB_LOG_DIR)
# )

# # Define callback for keras
# tensor_board_callback = keras.callbacks.TensorBoard(log_dir = TB_LOG_DIR,
#                                          histogram_freq=0,
#                                          write_graph=True,
#                                          write_images=True)


# # Get link to tensorboard
# get_ipython().system_raw('./ngrok http 6006 &')
# ! curl -s http://localhost:4040/api/tunnels | python3 -c \
#     "import sys, json; print(json.load(sys.stdin)['tunnels'][0]['public_url'])"
