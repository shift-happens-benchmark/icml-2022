"""Create papers.rst file with accepted contributions."""

import pathlib
import textwrap

import pandas as pd


def format_for_website(line):
    """Formatting function for pd.DataFrame.apply."""

    full_submission = "Full submission" in str(
        line.submission_type
    ) or "Full submission" in str(line.track)
    if full_submission:
        if line.decision == "Accept (Contributed Talk)":
            line["presentation_type"] = "talk"
        elif line.decision == "Accept (Poster)":
            line["presentation_type"] = "poster"
        else:
            raise ValueError(line.decision)
        line["category"] = "full"
    else:
        line["presentation_type"] = "poster"
        line["category"] = "abstract"

    line["authors"] = line["authors"].replace("|", ", ")
    line["formatted"] = "{title}.\n\t{authors}.".format(**line)
    return line


def filter_and_format(df, **kwargs):
    for key, value in kwargs.items():
        df = df[df[key] == value]
    return "\n\n".join(df["formatted"].sort_values())


def render_text(filename):

    df = pd.read_csv(filename)
    df = df[df.decision.apply(lambda v: "Accept" in v)]
    df = df.apply(format_for_website, axis=1)

    template = textwrap.dedent(
        """
  Accepted Papers
  ===============
  
  
  Track 1: Benchmarks, Datasets and Metrics
  -----------------------------------------

  Contributed Talks
  ~~~~~~~~~~~~~~~~~

  {talks}

  Posters
  ~~~~~~~

  {posters}

  Track 2: Extended abstracts
  ---------------------------

  {abstracts}

  """
    )

    page = template.format(
        talks=filter_and_format(df, category="full", presentation_type="talk"),
        posters=filter_and_format(df, category="full", presentation_type="poster"),
        abstracts=filter_and_format(df, category="abstract"),
    )

    with (pathlib.Path(__file__).parent / "papers.rst").open("w") as fh:
        print(page, file=fh)


def main():
    # TODO(stes) pull data from OpenReview as soon as papers are public.
    filename = pathlib.Path("/tmp/Shift_Happens_2022_paper_status.csv")
    if filename.exists():
        render_text(filename)
    else:
        print("Could not find the submission metadata. If you are a PC and want to")
        print("update the website, please download the data from")
        print(
            "https://openreview.net/group?id=ICML.cc/2022/Workshop/Shift_Happens/Program_Chairs#paper-status"
        )
        print("and place the file under /tmp, then re-run.")


if __name__ == "__main__":
    main()
