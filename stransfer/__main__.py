import click

@click.command()
@click.argument('style-image')
@click.argument('content-image')
def cli(style_image, content_image):
    print(123123)

if __name__ == "__main__":
    # cli(**{}) # suppress warning

    from stransfer import network
    print(network)