# ANSI color codes
_COLORS = {
    # foreground Colors
    'BL': '\033[30m',	    # Black
    'RD': '\033[91m',	    # Red
    'GR': '\033[92m',	    # Green
    'YW': '\033[93m',	    # Yellow
    'BU': '\033[94m',	    # Blue
    'MG': '\033[95m',	    # Magenta
    'CY': '\033[96m',	    # Cyan
    'WH': '\033[97m',	    # White
    'GY': '\033[90m',       # Gray
    'PR': '\033[38;5;129m', # Purple

    # light foreground colors
    'RD_L': '\033[31m',     # Light Red
    'GR_L': '\033[32m',	    # Light Green
    'YL_L': '\033[33m',	    # Light Yellow
    'BU_L': '\033[34m',	    # Light Blue
    'MG_L': '\033[35m',	    # Light Magenta
    'CY_L': '\033[36m',	    # Light Cyan

    # background Colors
    'BL_BG': '\033[40m',    # Background Black
    'RD_BG': '\033[41m',	# Background Red
    'GR_BG': '\033[42m',	# Background Green
    'YW_BG': '\033[43m',	# Background Yellow
    'BU_BG': '\033[44m',	# Background Blue
    'MG_BG': '\033[45m',	# Background Magenta
    'CY_BG': '\033[46m',	# Background Cyan
    'WH_BG': '\033[47m',	# Background White

    # text Styles
    'B': '\033[1m',         # Bold
    'U': '\033[4m',         # Underline
    'REV': '\033[7m',	    # Reversed
    'DM': '\033[2m',	    # Dim
}

R = f'\033[0m'

# generate color functions dynamically
for color, code in _COLORS.items():
    if color != 'reset':
        globals()[color] = lambda text, code = code: f"{code}{text}{R}"

__all__ = list(_COLORS.keys()) + ['R']