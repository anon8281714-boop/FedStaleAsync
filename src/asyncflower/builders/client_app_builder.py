from flwr.client import ClientApp
from flwr.common.context import Context
from asyncflower.model.model import CustomModel
from asyncflower.client import CustomClient
from asyncflower.data.preprocessing import load_client_data

def build_client_app(
    dataset: str,
    dataset_dir: str,
    batch_size: int, 
    seed: int, 
    model_architecture: str, 
    device: str = "cpu"
) -> ClientApp:
    def client_fn(context: Context):
        cid = str(context.node_config["partition-id"])

        train_loader, val_loader = load_client_data(
            cid = cid, 
            data_dir = dataset_dir,
            dataset = dataset,
            batch_size = batch_size,
            seed = seed
        )

        model = CustomModel(model = model_architecture)

        client = CustomClient(
            cid = cid, 
            train_loader = train_loader, 
            val_loader = val_loader, 
            model = model, 
            device = device,
        )

        return client

    client_app = ClientApp(client_fn = client_fn)

    return client_app