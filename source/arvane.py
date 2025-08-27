class ArvaneServer:
    def __init__(self):
        self.arvane_configuration = None
        
        self.net_connection = None
        self.recon_predictor = None
        self.extract_predictor = None
        self.predictor_analyzer = None

        self.current_result = None

        pass

    # return current status
    @property
    def status(self):
        return None