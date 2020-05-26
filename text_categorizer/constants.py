import signal

stop_signals = [
    #signal.SIGINT,      # SIGINT is sent by CTRL-C.
    signal.SIGTERM,     # SIGTERM is sent by Docker on CTRL-C or on a call to 'docker stop'.
]
random_state = 42
