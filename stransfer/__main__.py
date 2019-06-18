import click
import colored_traceback

from stransfer import clis


@click.group(commands={
    'video_st': clis.video_st.cli,
    'fast_st': clis.fast_st.cli,
    'gatys_st': clis.gatys_st.cli,
})
def cli():
    """
    Style Transfer
    """
    pass


if __name__ == "__main__":
    colored_traceback.add_hook()
    cli(**{}, prog_name='stransfer')  # suppress warning
