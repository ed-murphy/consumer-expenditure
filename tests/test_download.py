from cex.download import interview_url


def test_interview_url():
    assert interview_url(2024) == "https://www.bls.gov/cex/pumd/data/csv/intrvw24.zip"
    assert interview_url(2019) == "https://www.bls.gov/cex/pumd/data/comma/intrvw19.zip"
