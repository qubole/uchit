import click


@click.group()
@click.option("--verbose", is_flag=True, help="Will print verbose messages.")
@click.pass_context
def main(ctx, verbose):
    ctx.obj["verbose"] = verbose


@main.command()
@click.argument("config_json")
@click.pass_context
def spark(ctx, config_json):
    click.echo("Auto Tuning Spark")
    v_opt = "ON" if ctx.obj["verbose"] else "OFF"
    click.echo("verbose: " + v_opt)


def start():
    main(obj={})


if __name__ == "__main__":
    start()
