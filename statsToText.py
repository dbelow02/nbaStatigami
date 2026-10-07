#Takes the stat line ouputs and passes to chat gpt to pick a top 3 and write brief blurb about each stat line.

#Thinking using Google AI Studio is best because of it's free tier.


from dotenv import load_dotenv
import os
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
import datetime
import re

load_dotenv()  #reads .env file into environment variables

#Classes for pydantic to enfore on gemini output:
class TweetItem(BaseModel):
    player: str = Field(description="Name of the NBA player")
    tweet: str = Field(description="Tweet text under 280 characters")

class TweetList(BaseModel):
    posts: list[TweetItem]


API_KEY = os.getenv("GEMINI_API_KEY")

SYSTEM_PROMPT = """
You are a sharp, witty sports analytics account on Twitter. You surface daily rare NBA 
stat lines and write punchy posts about them. You have a dry, occasionally sarcastic sense of 
humor — especially when a stat line is more embarrassing than impressive.

You will be given a list of player performances from recent NBA games with:
- The player's name and rarity score
- The key stats that made it rare
- A SQL WHERE clause showing the exact thresholds (e.g. points >= 35 AND steals >= 3)
- How many times it's been achieved historically
- Who last achieved it and when

Your job:
1. Pick the 3 most interesting performances. Prioritize:
   - Genuinely impressive rare feats (high score, few historical occurrences)
   - Surprising players exceeding expectations
   - Hilariously bad stat lines that are rare in a bad way (e.g. -14 plus/minus, 0 points)
   - Interesting historical drought (last done years ago, or only done by legends)

2. For each, write one tweet under 280 characters that:
   - Names the player
   - VARY the structure of each tweet — never start two tweets the same way, and never 
     default to opening with "Only X times in NBA history". Lead with the player, the stat line, 
     the historical context, or a punchline — mix it up so each tweet has a distinct feel.
     Examples of varied openers:
       "SGA tonight: 35 pts, 9 ast, 3 stl, 1 blk. Only 73 times in NBA history."
       "73 times. That's how often someone has put up 35+ pts, 9+ ast, 3+ stl and 1+ blk. SGA just made it 74."
       "De'Aaron Fox scored 15, dished 5 assists, had 3 steals, and grabbed exactly 0 rebounds. Classic Fox."
   - If a statistic has been achieved a surprisingly scarce number of times,
     include how many times this specific combination has been achieved in NBA history,
     WITH the stat thresholds as context. e.g. "only 73 times has someone put up 35+ pts, 9+ ast, 
     3+ stl and 1+ blk in NBA history" — not just "only 73 times in NBA history"
   - Always leads with the specific stat thresholds naturally translated from the WHERE clause.
     The thresholds ARE the story — e.g. "35+ pts, 9+ ast, 3+ stl, 1+ blk" not just "great game".
     For <= thresholds, frame them naturally: "just 4 points", "0 rebounds", "-14 plus/minus"
   - If the last player to achieve it is notable (a legend, a funny contrast, or the same player),
     include it: "last done by LeBron in 2019" or "Wemby did this two days ago too"
   - Is sarcastic/funny if the stat line is bad, celebratory if genuinely impressive
   - Feels human, not generated!

3. The historical count MUST be paired with the stat context. Never write "X times in NBA history" 
   without specifying what combination of stats you're referring to.

Return ONLY raw JSON, with no markdown code fences, no ```json wrapper, and no preamble or explanation.
A well formatted output would be the following:
{
    "posts": [
        {
            "player": "player name",
            "tweet": "tweet text under 280 chars"
        }
    ]
}
"""

defaultinput="""Shai Gilgeous-Alexander scored 7.5946236075307585
With columns: ['points', 'steals', 'assists', 'blocks']
         points  steals  assists  blocks
1674732    35.0     3.0      9.0     1.0
With a SQL where clause of WHERE points >= 35.0 AND steals >= 3.0 AND assists >= 9.0 AND blocks >= 1.0
This statline was last achieved by Nikola Jokic on 2026-04-06 21:00:00. (achieved 73 times historically)

Stephon Castle scored 6.993291348044098
With columns: ['threePointersPercentage', 'points', 'assists', 'reboundsOffensive', 'turnovers']
         threePointersPercentage  points  assists  reboundsOffensive  turnovers
1674743                      0.0    16.0      6.0                4.0        6.0
With a SQL where clause of WHERE threePointersPercentage <= 0.0 AND points >= 16.0 AND assists >= 6.0 AND reboundsOffensive >= 4.0 AND turnovers >= 6.0
This statline was last achieved by Paolo Banchero on 2026-04-12 18:00:00. (achieved 131 times historically)

Julian Champagnie scored 6.714342362364311
With columns: ['plusMinusPoints', 'points', 'threePointersMade', 'steals']
         plusMinusPoints  points  threePointersMade  steals
1674741             16.0    20.0                6.0     1.0
With a SQL where clause of WHERE plusMinusPoints >= 16.0 AND points >= 20.0 AND threePointersMade >= 6.0 AND steals >= 1.0
This statline was last achieved by Miles McBride on 2026-05-10 15:30:00. (achieved 334 times historically)

Jaylin Williams scored 6.567869342674685
With columns: ['reboundsTotal', 'plusMinusPoints', 'points', 'assists', 'reboundsOffensive']
         reboundsTotal  plusMinusPoints  points  assists  reboundsOffensive
1674735           10.0             10.0    11.0      4.0                3.0
With a SQL where clause of WHERE reboundsTotal >= 10.0 AND plusMinusPoints >= 10.0 AND points >= 11.0 AND assists >= 4.0 AND reboundsOffensive >= 3.0
This statline was last achieved by Isaiah Hartenstein on 2026-05-26 20:30:00. (achieved 488 times historically)

Cason Wallace scored 6.537665999341959
With columns: ['points', 'threePointersMade', 'steals', 'blocks']
         points  threePointersMade  steals  blocks
1674731    17.0                5.0     2.0     1.0
With a SQL where clause of WHERE points >= 17.0 AND threePointersMade >= 5.0 AND steals >= 2.0 AND blocks >= 1.0
This statline was last achieved by Alex Caruso on 2026-05-18 20:30:00. (achieved 474 times historically)

Isaiah Hartenstein scored 6.419886984925183
With columns: ['plusMinusPoints', 'freeThrowsPercentage', 'steals', 'blocks', 'points']
         plusMinusPoints  freeThrowsPercentage  steals  blocks  points
1674730            -14.0                   1.0     1.0     0.0     7.0
With a SQL where clause of WHERE plusMinusPoints <= -14.0 AND freeThrowsPercentage >= 1.0 AND steals >= 1.0 AND blocks >= 0.0 AND points >= 7.0
This statline was last achieved by Jared McCain on 2026-05-28 20:30:00. (achieved 1346 times historically)

Victor Wembanyama scored 6.415139251132091
With columns: ['points', 'blocks', 'steals', 'threePointersMade', 'plusMinusPoints']
         points  blocks  steals  threePointersMade  plusMinusPoints
1674742    22.0     1.0     1.0                3.0              7.0
With a SQL where clause of WHERE points >= 22.0 AND blocks >= 1.0 AND steals >= 1.0 AND threePointersMade >= 3.0 AND plusMinusPoints >= 7.0
This statline was last achieved by Victor Wembanyama on 2026-05-28 20:30:00. (achieved 563 times historically)

Devin Vassell scored 6.236430202097024
With columns: ['steals', 'freeThrowsPercentage', 'points', 'reboundsTotal', 'blocks']
         steals  freeThrowsPercentage  points  reboundsTotal  blocks
1674740     2.0                   1.0    11.0            6.0     0.0
With a SQL where clause of WHERE steals >= 2.0 AND freeThrowsPercentage >= 1.0 AND points >= 11.0 AND reboundsTotal >= 6.0 AND blocks >= 0.0
This statline was last achieved by Victor Wembanyama on 2026-05-28 20:30:00. (achieved 1835 times historically)

Jared McCain scored 6.219077857901409
With columns: ['plusMinusPoints', 'points', 'steals', 'threePointersMade', 'blocks', 'reboundsTotal']
         plusMinusPoints  points  steals  threePointersMade  blocks  reboundsTotal
1674734             -8.0    12.0     1.0                2.0     0.0            1.0
With a SQL where clause of WHERE plusMinusPoints <= -8.0 AND points >= 12.0 AND steals >= 1.0 AND threePointersMade >= 2.0 AND blocks >= 0.0 AND reboundsTotal <= 1.0
This statline was last achieved by Donovan Mitchell on 2026-05-23 20:00:00. (achieved 608 times historically)

Chet Holmgren scored 6.0283407041487305
With columns: ['steals', 'blocks', 'assists', 'points', 'plusMinusPoints']
         steals  blocks  assists  points  plusMinusPoints
1674729     2.0     2.0      0.0     4.0             -3.0
With a SQL where clause of WHERE steals >= 2.0 AND blocks >= 2.0 AND assists <= 0.0 AND points <= 4.0 AND plusMinusPoints <= -3.0
This statline was last achieved by Mitchell Robinson on 2026-03-15 20:00:00. (achieved 136 times historically)

Luke Kornet scored 6.013531178426986
With columns: ['fieldGoalsPercentage', 'freeThrowsPercentage', 'blocks', 'efgPct']
         fieldGoalsPercentage  freeThrowsPercentage  blocks  efgPct
1674746                   0.0                   1.0     1.0     0.0
With a SQL where clause of WHERE fieldGoalsPercentage <= 0.0 AND freeThrowsPercentage >= 1.0 AND blocks >= 1.0 AND efgPct <= 0.0
This statline was last achieved by Isaiah Hartenstein on 2026-05-18 20:30:00. (achieved 1060 times historically)

De'Aaron Fox scored 6.012670038410899
With columns: ['steals', 'points', 'reboundsTotal', 'assists']
         steals  points  reboundsTotal  assists
1674744     3.0    15.0            0.0      5.0
With a SQL where clause of WHERE steals >= 3.0 AND points >= 15.0 AND reboundsTotal <= 0.0 AND assists >= 5.0
This statline was last achieved by James Harden on 2026-05-11 20:00:00. (achieved 179 times historically)

Keldon Johnson scored 5.8998923308982505
With columns: ['steals', 'points', 'plusMinusPoints', 'blocks', 'threePointersMade', 'reboundsOffensive']
         steals  points  plusMinusPoints  blocks  threePointersMade  reboundsOffensive
1674747     1.0    11.0              6.0     0.0                2.0                2.0
With a SQL where clause of WHERE steals >= 1.0 AND points >= 11.0 AND plusMinusPoints >= 6.0 AND blocks >= 0.0 AND threePointersMade >= 2.0 AND reboundsOffensive >= 2.0
This statline was last achieved by Victor Wembanyama on 2026-05-28 20:30:00. (achieved 1005 times historically)

Alex Caruso scored 5.68977523866695
With columns: ['blocks', 'points', 'assists', 'plusMinusPoints', 'reboundsTotal']
         blocks  points  assists  plusMinusPoints  reboundsTotal
1674733     1.0    12.0      4.0             -5.0            5.0
With a SQL where clause of WHERE blocks >= 1.0 AND points >= 12.0 AND assists >= 4.0 AND plusMinusPoints <= -5.0 AND reboundsTotal >= 5.0
This statline was last achieved by Stephon Castle on 2026-05-22 20:30:00. (achieved 753 times historically)

Dylan Harper scored 5.6122337057348375
With columns: ['plusMinusPoints', 'points', 'reboundsTotal', 'reboundsOffensive', 'threePointersMade']
         plusMinusPoints  points  reboundsTotal  reboundsOffensive  threePointersMade
1674745              9.0    12.0            7.0                3.0                2.0
With a SQL where clause of WHERE plusMinusPoints >= 9.0 AND points >= 12.0 AND reboundsTotal >= 7.0 AND reboundsOffensive >= 3.0 AND threePointersMade >= 2.0
This statline was last achieved by Victor Wembanyama on 2026-05-28 20:30:00. (achieved 597 times historically)

Luguentz Dort scored 5.475014913208062
With columns: ['plusMinusPoints', 'blocks', 'reboundsTotal', 'points', 'threePointersMade', 'steals']
         plusMinusPoints  blocks  reboundsTotal  points  threePointersMade  steals
1674728             -9.0     0.0            1.0     3.0                1.0     0.0
With a SQL where clause of WHERE plusMinusPoints <= -9.0 AND blocks >= 0.0 AND reboundsTotal <= 1.0 AND points <= 3.0 AND threePointersMade >= 1.0 AND steals >= 0.0
This statline was last achieved by Carter Bryant on 2026-05-22 20:30:00. (achieved 819 times historically)"""

def strip_code_fence(text: str) -> str:
    """Return the JSON inside a ```json ... ``` block, or the text as-is if there's no block."""
    m = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    return m.group(1) if m else text.strip()

def getLLMText(inputStatlines):

    retry_options = types.HttpRetryOptions(
        attempts=5,                  # Try up to 5 times before failing
        initial_delay=2.0,           # Wait 2 seconds before the first retry
        max_delay=60.0,              # Cap the wait time between retries at 60 seconds
        http_status_codes=[408, 429, 500, 502, 503, 504] # HTTP status codes to retry
    )

    client = genai.Client(api_key=API_KEY,http_options=types.HttpOptions(retry_options=retry_options))


    starttime = datetime.datetime.now()

    interaction = client.interactions.create(
        model="gemini-2.5-flash",
        input=inputStatlines,
        system_instruction=SYSTEM_PROMPT,
        response_format={
            "type": "text",
            "mime_type": "application/json",
            "schema": TweetList.model_json_schema()
        }
    )

    endtime = datetime.datetime.now()
    print(f"It took {endtime-starttime} for the LLM to process the request.")



    try:
        # Use Pydantic to validate and load the JSON string directly into Python objects
        parsed_tweets = TweetList.model_validate_json(strip_code_fence(interaction.output_text))
        
        lines = []
        for post in parsed_tweets.posts:
            lines.append(f"Player: {post.player}")
            lines.append(f"Tweet: {post.tweet}")
            lines.append("-" * 50)
        if len(lines) == 0:
            raise ValueError("Lines is empty!")
        return "\n".join(lines), parsed_tweets.posts
            
    except Exception as e:
        print(f"Failed to parse structured response: {e}")
        print("Raw Output was:")
        print(interaction.output_text)
        return None, None

