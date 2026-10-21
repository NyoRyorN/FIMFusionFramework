from matplotlib import font_manager, rcParams


def setPltFont():
    font_manager.fontManager.addfont("src/utils/Arial.ttf")
    rcParams['font.family'] = 'Arial'
