from src.core import MDR
from src.files_management import PageMeta

all_metas = PageMeta.get_all()


def download_raw(page_meta: PageMeta) -> None:
    pass


def cleanup_html(page_meta: PageMeta) -> None:
    pass


def persist_named_tags_html(page_meta: PageMeta, mdr: MDR) -> None:
    # todo remember to also markup the data records
    pass


def color_html(page_meta: PageMeta, mdr: MDR) -> None:
    pass


def persist_data_regions(page_meta: PageMeta, mdr: MDR) -> None:
    pass


def persist_data_records(page_meta: PageMeta, mdr: MDR) -> None:
    pass


def persist_mdr_parameters(page_meta: PageMeta, mdr: MDR) -> None:
    pass
