# add module doc tring to say that this is a modules with utility functions
"""
This module contains utility functions for the StreamPort application.
"""

import chardet


def get_file_encoding(file):
    """
    Detect the character encoding of a file.
    
    Args:
        file (str): Path to the file. Must properly handle escape characters or be a rawstring r"path".
    
    Returns:
        character_encoding (str): The encoding (e.g UTF-8/ASCII) that the file is saved in. 
    """
    
    with open(file, "rb") as f:
        rawdata = f.read()
    decoding_result = chardet.detect(rawdata)
    character_encoding = decoding_result["encoding"]

    return character_encoding
