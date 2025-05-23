import asyncio
import aiohttp
import nest_asyncio
import pandas as pd

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

async def fetch_fasta(session, uniprot_id):
    fasta_url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
    async with session.get(fasta_url) as response:
        if response.status != 200:
            print(f"Error fetching sequence for {uniprot_id}")
            return None
        text = await response.text()
        return "".join(text.splitlines()[1:])

async def fetch_json(session, uniprot_id):
    json_url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json"
    async with session.get(json_url) as response:
        if response.status != 200:
            print(f"Error fetching JSON data for {uniprot_id}")
            return None
        data = await response.json()
        transmembrane_domains = [
            (feature['location']['start']['value'], feature['location']['end']['value'])
            for feature in data.get("features", [])
            if feature.get("type") == "Transmembrane"
        ]
        return transmembrane_domains

async def fetch_data(uniprot_id):
    async with aiohttp.ClientSession() as session:
        fasta = await fetch_fasta(session, uniprot_id)
        transmembrane = await fetch_json(session, uniprot_id)
        transmembrane = transmembrane[0]
        print(f"Fetched {uniprot_id}")
        return fasta, transmembrane

async def main(clinvar_ss):
    results = await asyncio.gather(*[
        fetch_data(uniprot_id) for uniprot_id in clinvar_ss['seq_id']
    ])
    clinvar_ss['Sequence'], clinvar_ss['Transmembrane'] = zip(*results)

# Example usage
await main(clinvar_ss)

print(clinvar_ss)
