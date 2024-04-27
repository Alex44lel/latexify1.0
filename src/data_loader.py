from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
import torch

# TODO: this class should load the dataset directly from pkl file
# instead relying on external dataframe
#
# The `__init__` method should take in a tokenizer class.
# See `LMDataset` (project 3) for more details.


class CustomLatexDataset(Dataset):
    def __init__(self, df, y, split, tokenizer):
        assert split in {"train", "test", "validation"}

        if split == "train":
            self.start_split = 0
            self.end_split = 114405
        elif split == "test":
            self.start_split = 114406
            self.end_split = 114405 + 14280
        else:  # validation
            self.start_split = 114405 + 14280 + 1
            self.end_split = 114405 + 14280 + 14280

        self.split = split
        self.image_df = df.reset_index(drop=True)
        self.imgs_y = y
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.image_df.iloc[self.start_split: self.end_split])

    def __getitem__(self, idx):
        # print(idx)
        img_label = self.image_df.iloc[self.start_split + idx]["word2id"]
        name = self.image_df.iloc[self.start_split + idx]["image"]

        tok_label = self.tokenizer.encode(img_label)

        if not self.tokenizer.use_gpt:
            y = tok_label[1:]
            x = tok_label[:-1]
        # print(name)
        else:
            y = tok_label[:]
            x = tok_label[:]

        image = self.imgs_y[name]
        to_tensor = ToTensor()
        image = to_tensor(image)

        return image, x, y


def get_data_loaders(df_combined, y_combined, split, tokenizer, batch_size=56):
    current_dataset = CustomLatexDataset(
        df_combined, y_combined, split, tokenizer)

    current_dataset_loader = DataLoader(
        current_dataset, batch_size=batch_size, shuffle=True
    )

    return current_dataset_loader
