import click
from stransfer import img_utils, network
from tqdm import tqdm


def run_static_style_transfer(style_path, content_path, steps=220):

    # TODO remove these 2
    content_path = "data/dancing.jpg"
    style_path = "data/picasso.jpg"
    

    style_image = img_utils.image_loader(style_path)
    content_image = img_utils.image_loader(content_path)

    # clamp content image before creating network
    content_image.data.clamp_(0, 1)
    st_net = network.StyleNetwork(style_image, content_image)

    optimizer = network.get_content_optimizer(content_image)

    for step in tqdm(range(steps)):
        def closure():
        # clamp content image in place each step
        content_image.data.clamp_(0, 1)

        optimizer.zero_grad()

        # pass content image through net
        st_net(content_image)

        # get losses
        style_loss = st_net.get_total_current_style_loss()
        content_loss = st_net.get_total_current_content_loss() 

        total_loss = style_loss + content_loss
        
        print(total_loss)
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

    run_static_style_transfer(1,1,20)