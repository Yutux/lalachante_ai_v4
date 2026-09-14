import csv
from tqdm import tqdm
from src.config import MB_AREA_ARCHIVE, MB_AREAS_CSV
from src.utils.musicbrainz_tar import iter_json_dump


def joined_codes(record):
    codes = []
    for key in ["iso-3166-1-codes", "iso-3166-2-codes", "iso-3166-3-codes"]:
        codes.extend(str(v) for v in (record.get(key) or []))
    return "|".join(dict.fromkeys(codes))


def main():
    MB_AREAS_CSV.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "area_mbid", "area_name", "area_sort_name", "area_type",
        "area_type_id", "iso_codes", "disambiguation", "begin", "end", "ended",
    ]
    count = 0
    with MB_AREAS_CSV.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for record in tqdm(iter_json_dump(MB_AREA_ARCHIVE, "area"), desc="MusicBrainz areas"):
            life = record.get("life-span") or {}
            writer.writerow({
                "area_mbid": record.get("id"),
                "area_name": record.get("name"),
                "area_sort_name": record.get("sort-name"),
                "area_type": record.get("type"),
                "area_type_id": record.get("type-id"),
                "iso_codes": joined_codes(record),
                "disambiguation": record.get("disambiguation"),
                "begin": life.get("begin"),
                "end": life.get("end"),
                "ended": life.get("ended"),
            })
            count += 1
    print(f"{count:,} areas exportées vers {MB_AREAS_CSV}")


if __name__ == "__main__":
    main()
