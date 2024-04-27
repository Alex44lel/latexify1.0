import torch
import torch.optim as optim

from .encoders.resnet import resnet50, resnet34, resnet101, resnet18, resnet152
from .encoders.convnext import (
    convnext_tiny,
    convnext_small,
    convnext_base,
    convnext_large,
)
from .encoders.swin import swin_v2_t, swin_v2_s, swin_v2_b
from .decoders.transformer import gpt, transformer
from .models import LatexifyModel
from .tokenizer import Tokenizer
from .analysis import BLEUScore, MulticlassAccuracy, Perplexity


# Construct our model and tokenizer based on the given encoder, decoder
# 1. Different encoder requires different embedding dim for the decoder
# 2. Tokenizer should behave differently based on the decoder we choose.
def get_model(
    encoder_name: str,
    decoder_name: str,
    encoder_args: dict,
    decoder_args: dict,
    tokenizer_args: dict,
):
    encoder = None
    decoder = None
    embed_dim = None
    tokenizer = Tokenizer(
        **{"use_gpt": decoder_name == "gpt", **tokenizer_args}
    )

    if encoder_name == "resnet18":
        encoder = resnet18(**encoder_args)
        embed_dim = 512
    elif encoder_name == "resnet34":
        encoder = resnet34(**encoder_args)
        embed_dim = 512
    elif encoder_name == "resnet50":
        encoder = resnet50(**encoder_args)
        embed_dim = 2048
    elif encoder_name == "resnet101":
        encoder = resnet101(**encoder_args)
        embed_dim = 2048
    elif encoder_name == "resnet152":
        encoder = resnet152(**encoder_args)
        embed_dim = 2048
    elif encoder_name == "convnext_tiny":
        encoder = convnext_tiny(**encoder_args)
        embed_dim = 768
    elif encoder_name == "convnext_small":
        encoder = convnext_small(**encoder_args)
        embed_dim = 768
    elif encoder_name == "convnext_base":
        encoder = convnext_base(**encoder_args)
        embed_dim = 1024
    elif encoder_name == "convnext_large":
        encoder = convnext_large(**encoder_args)
        embed_dim = 1024
    elif encoder_name == "swin_v2_t":
        encoder = swin_v2_t(**encoder_args)
        embed_dim = 768
    elif encoder_name == "swin_v2_s":
        encoder = swin_v2_s(**encoder_args)
        embed_dim = 768
    elif encoder_name == "swin_v2_b":
        encoder = swin_v2_b(**encoder_args)
        embed_dim = 1024

    if decoder_name == "gpt":
        decoder = gpt(
            **{
                "n_embd": embed_dim,
                "vocab_size": tokenizer.get_vocab_size(),
                **decoder_args,
            }
        )  # allow decoder_args to overwrite `n_embed`, but by default, it should work well
    elif decoder_name == "transformer":
        decoder = transformer(
            **{
                "n_embd": embed_dim,
                "vocab_size": tokenizer.get_vocab_size(),
                **decoder_args,
            }
        )

    return LatexifyModel(encoder, decoder, encoder_name + "-" + decoder_name), tokenizer


# Reference
# - https://nlp.seas.harvard.edu/annotated-transformer/#inference
def train(train_loader, test_loader, model, tokenizer, num_epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.AdamW(model.parameters())

    for epoch in range(num_epochs):
        # train
        model.train()
        train_loss = 0
        train_acc = MulticlassAccuracy()
        train_per = Perplexity(ignore_index=-1)
        train_bleu = BLEUScore(n_gram=4)
        for images, x, y in train_loader:
            images = images.to(device)
            x = x.to(device)
            y = y.to(device)

            logits, loss = model(x, images, y)  # type: ignore
            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # ==== Analysis ====

            # accuracy
            # TODO: use correct y here
            _, predicted = torch.max(logits, dim=-1)
            train_acc.update(predicted.view(-1), y.view(-1))  # type: ignore

            # bleu: incorrect impl right now
            # TODO: use correct y here
            reference_corpus = [
                [tokenizer.decode_seq(y_seq)] for y_seq in y  # type: ignore
            ]
            candidate_corpus = [
                tokenizer.decode_seq(predicted_seq) for predicted_seq in predicted
            ]
            train_bleu.update(candidate_corpus, reference_corpus)

            # perplexity: lower -> better
            # TODO: use correct y here
            train_per.update(logits, y)  # type: ignore

        avg_loss = train_loss / len(train_loader)

        print(
            f"Train: Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, "
            f"Accuracy: {train_acc.compute():.4f}, BLEU: {
                train_bleu.compute():.4f}, "
            f"Perplexity: {train_per.compute():.4f}"
        )

        # test
        model.eval()
        test_loss = 0
        test_acc = MulticlassAccuracy()
        test_per = Perplexity(ignore_index=-1)
        test_bleu = BLEUScore(n_gram=4)
        with torch.inference_mode():
            for images, x, y in test_loader:
                images = images.to(device)
                x = x.to(device)
                y = y.to(device)

                # TODO: use correct x and y here
                logits, loss = model(x, images, y)  # type: ignore
                test_loss += loss.item()

                # ==== Analysis ====

                # accuracy
                # TODO: use correct y here
                _, predicted = torch.max(logits, dim=-1)
                test_acc.update(predicted.view(-1), y.view(-1))  # type: ignore

                # bleu: incorrect impl right now
                # TODO: use correct y here
                reference_corpus = [
                    # type: ignore
                    [tokenizer.decode_seq(y_seq)] for y_seq in y
                ]
                candidate_corpus = [
                    tokenizer.decode_seq(predicted_seq) for predicted_seq in predicted
                ]
                test_bleu.update(candidate_corpus, reference_corpus)

                # perplexity: lower -> better
                # TODO: use correct y here
                test_per.update(logits, y)  # type: ignore

        avg_loss = test_loss / len(test_loader)
        print(
            f"Test: Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, "
            f"Accuracy: {test_acc.compute():.4f}, BLEU: {
                test_bleu.compute():.4f}, "
            f"Perplexity: {test_per.compute():.4f}"
        )

    torch.save(model.state_dict(), f"./models/latexify-{model.name}.pth")
