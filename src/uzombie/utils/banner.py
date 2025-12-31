# src/uzombie/utils/banner.py
from .logger import console

def print_zombie_banner():
    # Toxic Slime Style Banner
    banner = """
██╗   ██╗███████╗ ██████╗ ███╗   ███╗██████╗ ██╗███████╗
██║   ██║╚══███╔╝██╔═══██╗████╗ ████║██╔══██╗██║██╔════╝
██║   ██║  ███╔╝ ██║   ██║██╔████╔██║██████╔╝██║█████╗  
██║   ██║ ███╔╝  ██║   ██║██║╚██╔╝██║██╔══██╗██║██╔══╝  
╚██████╔╝███████╗╚██████╔╝██║ ╚═╝ ██║██████╔╝██║███████╗
 ╚═════╝ ╚══════╝ ╚═════╝ ╚═╝     ╚═╝╚═════╝ ╚═╝╚══════╝
"""
    # Print the logo in bold green using Rich
    console.print(banner.strip(), style="bold green")
    # Print the divider line
    console.rule("[bold green]UZOMBIE v1.3[/]")