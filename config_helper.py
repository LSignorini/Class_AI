import json
import jsonschema
from jsonschema import exceptions as jsonschema_exceptions

from pathlib import Path
from types import SimpleNamespace


def check_and_get_configuration (filename : str, validation_filename : str) -> object:
    """
    Dato un file di configurazione json ed uno schema di validazione json,
    verifica che il primo soddisfi le condizioni dettate dal secondo. In
    caso di errore o mancata validazione ritorna None.

    Args:
        filename (str): Nome/Percorso del file di configurazione json.
        validation filename (str): Nome/Percorso del file di validazione json.

    Returns:
        object: Oggetto python strutturato a partire dal json di configurazione.
    """

    json_object = None
    # Indico dove trovare i json di configurazione e di verifica
    data_file = Path(filename)
    schema_file = Path(validation_filename)
    
    # Controllo che i file esistano e siano dei json.
    if (data_file.is_file() and schema_file.is_file() and
        data_file.suffix == '.json' and schema_file.suffix == '.json'):
        
        with open(Path(data_file)) as d:
            with open(Path(schema_file)) as s:
            
                # Carico i due json e utilizzo lo schema per validare il file di configurazione.
                data = json.load(d)
                schema = json.load(s)

                try:
                    jsonschema.validate(instance=data, schema=schema)
                except jsonschema_exceptions.ValidationError:
                    print(f'Json config file is not following schema rules.')
                    return json_object
                except jsonschema_exceptions.SchemaError:
                    print(f'Json config schema file is invalid.')
                    return json_object
                
    # A questo punto possiedo un file di configurazione json sintatticamente valido.
    # Passo il contenuto al synthetic builder.
    with open(Path(data_file)) as d:
        json_object = json.loads(d.read(), object_hook=lambda d: SimpleNamespace(**d))
        """ SimpleNamespace al posto di Config """

    return json_object