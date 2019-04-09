import click

@click.command()
@click.argument('style-source')
@click.argument('target-image')
def cli(style_source, target_image):
    print(123123)

if __name__ == "__main__":
    cli()