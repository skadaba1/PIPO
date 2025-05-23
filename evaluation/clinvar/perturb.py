import re
import ast
JXT_LENGTH = 8


def three_to_one(aa_code):
    aa_dict = {
        'Ala': 'A', 'Arg': 'R', 'Asn': 'N', 'Asp': 'D', 'Cys': 'C',
        'Gln': 'Q', 'Glu': 'E', 'Gly': 'G', 'His': 'H', 'Ile': 'I',
        'Leu': 'L', 'Lys': 'K', 'Met': 'M', 'Phe': 'F', 'Pro': 'P',
        'Ser': 'S', 'Thr': 'T', 'Trp': 'W', 'Tyr': 'Y', 'Val': 'V',
        'del': '',
    }
    return aa_dict.get(aa_code, 'X')  # 'X' for unknown codes

# Function to extract and mutate sequence
def mutate_and_extract(name, sequence, transmembrane):
    transmembrane = ast.literal_eval(transmembrane)
    # Find mutation info with regex
    match = re.search(r'p\.([A-Za-z]+)(\d+)([A-Za-z]+)', name)
    if match:
        original_aa, position, new_aa = match.groups()
        position = int(position) - 1  # Convert to 0-based index

        # Check if amino acid actually changes
        if original_aa != new_aa:
            # Apply mutation
            if(new_aa.lower() == 'ter'):
              mutated_seq = sequence[:position]
            elif(new_aa.lower() == 'dup'):
              mutated_seq = sequence[:position] + three_to_one(original_aa) + sequence[position + 1:]
            elif(new_aa.lower() == 'fs'):
              # try:
              #   mutated_seq, offset = parse_and_mutate_fs(name, sequence)
              #   tm_seq = mutated_seq[transmembrane[0]:transmembrane[1]+offset] + sequence[transmembrane[1]+offset:transmembrane[1]+offset+JXT_LENGTH]
              #   return tm_seq
              # except Exception as e:
              return None
            else:
              mutated_seq = sequence[:position] + three_to_one(new_aa) + sequence[position + 1:]
            tm_seq = mutated_seq[transmembrane[0]:transmembrane[1]] + mutated_seq[transmembrane[1]:transmembrane[1]+JXT_LENGTH]
            return tm_seq
    return None

def extract(name, sequence, transmembrane):
  transmembrane = ast.literal_eval(transmembrane)
  return sequence[transmembrane[0]:transmembrane[1]+JXT_LENGTH]