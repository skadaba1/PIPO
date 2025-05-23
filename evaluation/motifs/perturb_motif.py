# parse substrate seqs for intramembrane TMs
import re
motifs_of_interest = ['GXXXG'] #'GXGD', 'RSVLS', 'GXXXG', 'HEXXH', 'RSVLS', 'CGGX', 'CAAX', 'CVVX', 'CLLX', 'CIIX', 'CPPX']

def find_motifs(sequences, motif):
    # Convert the motif pattern, where X is any character
    motif_regex = motif.replace('X', '.')  # '.' in regex matches any character
    pattern = re.compile(motif_regex)

    result = []
    for seq in sequences:
        matches = [(m.start(), m.group()) for m in pattern.finditer(seq)]
        if matches:
            result.append((seq, matches))
    return result

def replace_randomly(string, target, replacement, num_to_replace):
    # Find all occurrences of the target substring
    occurrences = [i for i in range(len(string)) if string.startswith(target, i)]

    # Randomly sample indices to replace
    if num_to_replace > len(occurrences):
        raise ValueError("num_to_replace exceeds the number of available occurrences")

    indices_to_replace = random.sample(occurrences, num_to_replace)

    # Replace selected indices in the string
    result = list(string)
    for index in indices_to_replace:
        result[index:index+len(target)] = replacement

    return ''.join(result)

def select_random_not_in_motif(sequence, motif, roi):
    # Find all positions of the motif in the sequence
    motif_positions = [
        match.span() for match in re.finditer(motif.replace('X', '.'), sequence)
    ]

    # Create a set of all indices occupied by the motif
    motif_indices = set()
    for start, end in motif_positions:
        motif_indices.update(range(start, end))

    # Find all indices of 'G' in the sequence
    g_indices = [i for i, char in enumerate(sequence) if char == roi]

    # Filter out 'G' indices that are part of the motif
    valid_g_indices = [i for i in g_indices if i not in motif_indices]

    # If there are no valid 'G's, return None
    if not valid_g_indices:
        return None

    # Randomly choose a 'G' index from the valid ones
    return random.choice(valid_g_indices)