from dataclasses import dataclass

@dataclass(frozen=True)
class ThemeColors:
    # Font settings
    FONT_FAMILY: str = "Arial, sans-serif"

    # Plot backgrounds
    BACKGROUND_COLOR: str = "#0d0b0c"
    PAPER_BACKGROUND_COLOR: str = "#0d0b0c"

    # Text colors
    TEXT_PRIMARY: str = "#e8ebe8"
    TEXT_SECONDARY: str = "#01234a"
    TEXT_ON_DARK: str = "#ffffff"

    # Title styling
    TITLE_FONT_FAMILY: str = "Times New Roman, serif"
    TITLE_FONT_SIZE: int = 30
    TITLE_PRICE_COLOR: str = TEXT_PRIMARY

    # Bar colors
    BAR_CALL_COLOR: str = "#435744"
    BAR_PUT_COLOR: str = "#593636"

    # Line colors
    LINE_CALL_COLOR: str = "#75f542"
    LINE_PUT_COLOR: str = "#f54242"
    AVG_STRIKE_COLOR: str = "#565887"
    AVG_STRIKE_LINE_COLOR: str = "rgba(40,40,43,1)"

    # Secondary/tertiary line colors
    SECOND_CALL_LINE_COLOR: str = "#57f542"
    THIRD_CALL_LINE_COLOR: str = "#25f74f"
    SECOND_PUT_LINE_COLOR: str = "#d16262"
    THIRD_PUT_LINE_COLOR: str = "#d17b7b"

    # Border highlights (for extreme diffs)
    CALL_BORDER_HIGHLIGHT: str = "#29ff30"
    PUT_BORDER_HIGHLIGHT: str = "#ff2e2e"
    TRANSPARENT: str = "rgba(0,0,0,0)"

    # Marker styles
    MARKER_CALL_COLOR: str = "#75f542"
    MARKER_CALL_SYMBOL: str = "square"
    MARKER_PUT_COLOR: str = "#de3557"
    MARKER_PUT_SYMBOL: str = "square"

    # Marker sizing
    SIZE_FLOOR: int = 5
    SIZE_PEAK: int = 30

    # Annotation styles
    ANNOTATION_BG: str = "#515452"
    ANNOTATION_BORDER_COLOR: str = "#636363"
    ANNOTATION_FONT_SIZE: int = 12

    # Volume/premium highlights
    CALL_VOLUME_HIGHLIGHT: str = "#32a852"
    PUT_VOLUME_HIGHLIGHT: str = MARKER_PUT_COLOR

    # Current price line & annotation
    CURRENT_PRICE_LINE_COLOR: str = "#00dbf4"
    CURRENT_PRICE_ANNOTATION_BG: str = "#333333"
    CURRENT_PRICE_ANNOTATION_FONT_SIZE: int = 14

    # “Most Active” annotation backgrounds
    MOST_ACTIVE_UNUSUAL_BG_TOP: str = "#0d1138"
    MOST_ACTIVE_UNUSUAL_BG: str = "#2a1c63"
    MOST_ACTIVE_DEFAULT_BG: str = "#3b3b3b"

    # “Most Active” marker
    MOST_ACTIVE_MARKER_SYMBOL: str = "diamond"
    MOST_ACTIVE_MARKER_CALL_COLOR: str = "#32a852"
    MOST_ACTIVE_MARKER_PUT_COLOR: str = "#ff5e00"

    # Legend marker & text
    LEGEND_MARKER_COLOR: str = "#3b3b3b"
    LEGEND_MARKER_BORDER_COLOR: str = "#636363"
    LEGEND_TEXT_COLOR: str = "#10112e"

# Base Plotly layout template
BASE_LAYOUT = {
    "plot_bgcolor": ThemeColors.BACKGROUND_COLOR,
    "paper_bgcolor": ThemeColors.PAPER_BACKGROUND_COLOR,
    "showlegend": False,
    "legend": {
        "x": 0.5,
        "y": 1.10,
        "xanchor": "center",
        "yanchor": "top",
        "orientation": "h",
        "font": {"family": ThemeColors.FONT_FAMILY, "size": ThemeColors.ANNOTATION_FONT_SIZE, "color": ThemeColors.TEXT_PRIMARY},
    },
    "xaxis": {
        "title": "",
        "showgrid": False,
        "showline": True,
        "linecolor": "#444444",
        "linewidth": 1,
        "tickangle": 38,
        "tickfont": {"family": ThemeColors.FONT_FAMILY, "size": ThemeColors.ANNOTATION_FONT_SIZE + 4, "color": ThemeColors.TEXT_PRIMARY},
    },
    "yaxis": {"title": "", "showticklabels": False, "showgrid": False, "side": "right", "autorange": True},
    "yaxis2": {
        "title": "Strike",
        "title_font": {"family": ThemeColors.TITLE_FONT_FAMILY, "size": ThemeColors.TITLE_FONT_SIZE, "color": ThemeColors.TEXT_PRIMARY},
        "tickfont": {"family": ThemeColors.TITLE_FONT_FAMILY, "size": ThemeColors.TITLE_FONT_SIZE - 10, "color": ThemeColors.TEXT_PRIMARY},
        "side": "left",
        "overlaying": "y",
        "showline": False,
        "linecolor": "#444444",
        "linewidth": 0.5,
        "showgrid": True,
        "gridcolor": "rgba(136,136,136,0.10)",
        "zeroline": True,
        "zerolinecolor": "rgba(136,136,136,0.25)",
        "zerolinewidth": 0.5,
        "gridwidth": 0.5,
    },
    "barmode": "group",
}
