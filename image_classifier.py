from net_runner import NetRunner
from config_helper import check_and_get_configuration

if __name__ == "__main__":

    # Carica il file di configurazione, lo valida e ne crea un oggetto a partire dal json.
    cfg_obj = check_and_get_configuration ('./config.json', './config_schema.json')
    
    # Crea un oggetto runner con cui gestire la rete, il training ed il test
    runner = NetRunner (cfg_obj, True, train_percentage=0.001)

    #In caso di predict non verrà eseguito l'addestramento
    if cfg_obj.parameters.predict:
        runner.predict(cfg_obj.predict_parameters.path_image, cfg_obj.predict_parameters.path_model)
    
    else:

        #In caso di training abilitato, addestra la rete sui dati di addestramento
        if cfg_obj.parameters.train:
            runner.train(cfg_obj.parameters.show_preview)
        
        #In caso di test abilitato, valuta la rete sui dati di test
        if cfg_obj.parameters.test:
            runner.test(cfg_obj.parameters.show_preview)