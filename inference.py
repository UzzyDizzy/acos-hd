import pandas as pd
from openai import OpenAI
from tqdm import tqdm

from configs import (
    get_config,
    OPENAI_API_KEY
)

from data_filtering import (
    load_datasets
)

from prompt_stage1 import (
    load_few_shot_from_gold
)

from prompt_stage2 import (
    load_few_shot_stage2_from_gold
)

from stage1 import (
    generate_and_validate_stage1
)

from stage2 import (
    generate_and_validate_stage2
)

from repair_stage1 import (
    repair_stage1
)

from repair_stage2 import (
    repair_stage2
)


cfg=get_config()

client=OpenAI(
    api_key=OPENAI_API_KEY
)

##################################################
# load few-shot examples once
##################################################

_,gold_df=load_datasets(
    cfg.data
)

few_shot_s1=load_few_shot_from_gold(
    gold_df,
    cfg.pipeline.num_few_shot_examples
)

few_shot_s2=load_few_shot_stage2_from_gold(
    gold_df,
    cfg.pipeline.num_few_shot_examples
)


##################################################
# inference
##################################################

def predict_text(
    text:str
):

    #############################################
    # Stage 1
    #############################################

    s1_out,s1_vr,_,_=generate_and_validate_stage1(
        text,
        cfg,
        few_shot_s1,
        client
    )

    if s1_out is None:

        return {
            "text":text,
            "error":"stage1 failed"
        }


    if not s1_vr.valid:

        s1_out,s1_vr,_=repair_stage1(
            text,
            s1_out,
            s1_vr,
            cfg
        )

        if not s1_vr.valid:

            return {
                "text":text,
                "error":"stage1 repair failed"
            }


    #############################################
    # Stage 2
    #############################################

    s2_out,s2_vr,_,_=generate_and_validate_stage2(
        text,
        s1_out,
        cfg,
        few_shot_s2,
        client
    )


    if s2_out is None:

        return {
            "text":text,
            "error":"stage2 failed"
        }


    if not s2_vr.valid:

        s2_out,s2_vr,_=repair_stage2(
            text,
            s1_out,
            s2_out,
            s2_vr,
            cfg
        )

        if not s2_vr.valid:

            return {
                "text":text,
                "error":"stage2 repair failed"
            }


    #############################################
    # output
    #############################################

    return {

        "text":text,

        "aspect_target":
        s1_out.get(
            "aspect_target",
            ""
        ),

        "aspect_category":
        s1_out.get(
            "aspect_category",
            ""
        ),

        "opinion_span":
        s1_out.get(
            "opinion_span",
            ""
        ),

        "stance":
        s2_out.get(
            "stance",
            ""
        ),

        "explanation":
        s2_out.get(
            "explanation",
            ""
        )
    }



def predict_batch(
    texts
):

    outputs=[]

    for text in tqdm(
        texts,
        desc="Predicting"
    ):

        outputs.append(
            predict_text(
                text
            )
        )

    return pd.DataFrame(
        outputs
    )



##################################################
# Main
##################################################


    print(
        "Loading test.csv..."
    )

    df=pd.read_csv(
        "test.csv"
    )

    # assumes column name = text
    texts=df[
        "text"
    ].fillna(
        ""
    ).astype(
        str
    ).tolist()


    out=predict_batch(
        texts
    )


    result=pd.concat(
        [
            df.reset_index(
                drop=True
            ),
            out.drop(
                columns=["text"],
                errors="ignore"
            )
        ],
        axis=1
    )


    result.to_csv(
        "test_results.csv",
        index=False
    )

    print(
        "\nSaved -> test_results.csv"
    )

    print(
        result.head()
    )