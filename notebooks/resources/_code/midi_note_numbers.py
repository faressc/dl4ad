def midi_to_note(midi_number):
    """
    Convert a MIDI note number to a note string.
    
    Args:
        midi_number (int): MIDI note number (0-127)
        
    Returns:
        str: Note name with octave (e.g., "C4", "A#5")
        
    Examples:
        >>> midi_to_note(60)
        'C4'
        >>> midi_to_note(69)
        'A4'
        >>> midi_to_note(21)
        'A0'
    """
    if not 0 <= midi_number <= 127:
        raise ValueError(f"MIDI note number must be between 0 and 127, got {midi_number}")
    
    # Note names using sharps
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    # Calculate octave and note
    octave = (midi_number // 12) - 1
    note = note_names[midi_number % 12]
    
    return f"{note}{octave}"
