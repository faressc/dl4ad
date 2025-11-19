import numpy as np
from IPython.display import Audio



class SimpleSynth():
    def __init__(self, tempo=160, amplitude=0.1, sample_rate=44100, baroque_tuning=False):
        self.tempo = tempo
        self.amplitude = amplitude
        self.sample_rate = sample_rate
        self.baroque_tuning = baroque_tuning
        
        if baroque_tuning:
            # Baroque: A4 = 430 Hz with Kellner temperament
            self.a4_frequency = 430
            # Kellner temperament (1707) - cents deviation from equal temperament
            # Starting from C
            self.temperament = np.array([
                0,      # C
                -7.8,   # C#
                -3.9,   # D
                -7.8,   # D#
                -2.0,   # E
                0,      # F
                -7.8,   # F#
                -2.0,   # G
                -5.9,   # G#
                -3.9,   # A
                -5.9,   # A#
                -2.0    # B
            ]) / 100
        else:
            # Modern: A4 = 440 Hz with equal temperament
            self.a4_frequency = 440
            self.temperament = np.zeros(12)
    
    @staticmethod
    def asin(frequencies, time):
        return np.sin(2 * np.pi * frequencies.reshape(-1, 1) * time)
    
    def notes_to_frequencies(self, notes):
        notes = np.array(notes)
        
        # Calculate base frequency using equal temperament
        base_freq = 2 ** ((notes - 69) / 12) * self.a4_frequency
        
        # Apply temperament adjustments if using Baroque tuning
        if self.baroque_tuning:
            # Get the note within the octave (0-11, where 0=C)
            note_in_octave = (notes - 60) % 12  # 60 is middle C (C4)
            
            # Get temperament adjustment in cents for each note
            cents_adjustment = self.temperament[note_in_octave.astype(int)]
            
            # Apply adjustment: multiply by 2^(cents/1200)
            base_freq = base_freq * (2 ** (cents_adjustment / 1200))
        
        return base_freq

    def frequencies_to_samples(self, frequencies):
        note_duration = 60 / self.tempo # the tempo is measured in beats per minutes
        # To reduce click sound at every beat, we round the frequencies to try to
        # get the samples close to zero at the end of each note.
        frequencies = np.round(note_duration * frequencies) / note_duration
        n_samples = int(note_duration * self.sample_rate)
        time = np.linspace(0, note_duration, n_samples)
        sine_waves = self.asin(frequencies, time)
        # Removing all notes with frequencies <= 9 Hz (includes note 0 = silence)
        sine_waves *= (frequencies > 9.).reshape(-1, 1)
        return sine_waves.reshape(-1)

    def chords_to_samples(self, chords):
        freqs = self.notes_to_frequencies(chords)
        freqs = np.r_[freqs, freqs[-1:]] # make last note a bit longer
        merged = np.mean([self.frequencies_to_samples(melody)
                         for melody in freqs.T], axis=0)
        n_fade_out_samples = self.sample_rate * 60 // self.tempo # fade out last note
        fade_out = np.linspace(1., 0., n_fade_out_samples)**2
        merged[-n_fade_out_samples:] *= fade_out
        return self.amplitude * merged

    def play_chorale(self, choral_chords):
        samples = self.chords_to_samples(choral_chords)
        return display(Audio(samples, rate=self.sample_rate))
        
    def save_chorale(self, choral_chords, filepath):
        from scipy.io import wavfile
        samples = self.chords_to_samples(choral_chords)
        samples = (2**15 * samples).astype(np.int16)
        wavfile.write(filepath, self.sample_rate, samples)
