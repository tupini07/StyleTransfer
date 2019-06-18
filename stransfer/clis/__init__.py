import click

from stransfer.clis import video_st, fast_st, gatys_st


@click.group(commands={
    'video_st': video_st.video_st,
    'fast_st': fast_st.fast_st,
    'gatys_st': gatys_st.gatys_st,
})
def cli():
    """
    Style Transfer
    """
    pass
