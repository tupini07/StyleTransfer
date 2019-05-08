import click
from stransfer import img_utils, network


def run_static_style_transfer(style_path, content_path, steps=220):

    # TODO remove these 2
    content_path = "data/dancing.jpg"
    style_path = "data/picasso.jpg"
    

    style_image = img_utils.image_loader(style_path)
    content_image = img_utils.image_loader(content_path)

    st_net = network.StyleNetwork()

    optimizer = network.get_content_optimizer(content_image)

    for step in range(steps):
        # clamp content image in place
        content_image.data.clamp_(0, 1)

        optimizer.zero_grad()
        style_losses, content_losses = st_net(style_image, content_image)

        style_score = 0
        content_score = 0

        for sty_l in style_losses:
            # TODO for some reason the last style loss has no `.loss`
            # attribute. It is not layer specific since I've already tried changing
            # the style layer to other ones, and ever removing, but the last one always has
            # no loss
            # it might be that it never actually gets computed
            style_score += sty_l.loss

        for cont_l in content_losses:
            content_score += cont_l.loss

        total_loss = style_score + content_score
        total_loss.backward()

        optimizer.step()

    # TODO check if this is necessary
    content_image.data.clamp_(0, 1)

    return content_image


@click.command()
@click.argument('style-image')
@click.argument('content-image')
def cli(style_image, content_image):
    print(123123)


if __name__ == "__main__":
    # cli(**{}) # suppress warning

    run_static_style_transfer(1,1,1)