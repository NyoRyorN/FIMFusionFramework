def Tasks():
    """
    01-Sitting \n
    02-Upper \n
    03-SittingForward \n
    04-SittingDownwardRight \n
    05-SittingDownwardLeft \n
    06-Standing\n
    07-StandingForward\n
    08-StandingDownwardRight\n
    09-StandingDownwardLeft\n
    10-Squat\n
    11-Walking\n
    """
    return ["01-Sitting",
            "02-Upper",
            "03-SittingForward",
            "04-SittingDownwardRight",
            "05-SittingDownwardLeft",
            "06-Standing",
            "07-StandingForward",
            "08-StandingDownwardRight",
            "09-StandingDownwardLeft",
            "10-Squat",
            "11-Walking"
            ]


def plotTasks():
    """
    Static Sitting [Sit] \n
    Sitting Upper Limb Elevation [SitULE]\n
    Sitting Forward Reach [SitFR] \n
    Sitting Downward Right Reach [SitDRR] \n
    Sitting Downward Left Reach [SitDLR] \n
    Static Standing [Std]\n
    Standing Forward Reach [StdFR]\n
    Standing Downward Right Reach [StdDRR]\n
    Standing Downward Left Reach [StdDLR]\n
    Squat Exercise [Squat]\n
    Walking Forward [Walk]\n
    """
    return [
        "Sit",
        "Sit\nULE",
        "Sit\nFR",
        "Sit\nDRR",
        "Sit\nDLR",
        "Std",
        "Std\nFR",
        "Std\nDRR",
        "Std\nDLR",
        "Squat",
        "Walk"
    ]


def JPNTasks():
    return [
        "静止座位",
        "上肢挙上",
        "座位前方\nリーチ",
        "座位側方\nリーチ\n（右）",
        "座位側方\nリーチ\n（左）",
        "静止立位",
        "立位前方\nリーチ",
        "立位側方\nリーチ\n（右）",
        "立位側方\nリーチ\n（左）",
        "スクワット",
        "歩行"
    ]


def Joints():
    """
    SpineBase\n
    SpineMid \n
    Neck\n
    Head \n
    ShoulderLeft\n
    ElbowLeft\n
    WristLeft\n
    HandLeft\n
    ShoulderRight\n
    ElbowRight\n
    WristRight\n
    HandRight\n
    HipLeft\n
    KneeLeft\n
    AnkleLeft\n
    FootLeft\n
    HipRight\n
    KneeRight\n
    AnkleRight\n
    FootRight\n
    SpineShoulder\n
    HandTipLeft\n
    ThumbLeft\n
    HandTipRight\n
    ThumbRight\n
    """
    return ["SpineBase",
            "SpineMid",
            "Neck",
            "Head",
            "ShoulderLeft",
            "ElbowLeft",
            "WristLeft",
            "HandLeft",
            "ShoulderRight",
            "ElbowRight",
            "WristRight",
            "HandRight",
            "HipLeft",
            "KneeLeft",
            "AnkleLeft",
            "FootLeft",
            "HipRight",
            "KneeRight",
            "AnkleRight",
            "FootRight",
            "SpineShoulder",
            "HandTipLeft",
            "ThumbLeft",
            "HandTipRight",
            "ThumbRight"
            ]


def SimpleJoints():
    """
    SpineBase\n
    SpineMid \n
    Neck\n
    Head \n
    ShoulderLeft\n
    ElbowLeft\n
    WristLeft\n
    ShoulderRight\n
    ElbowRight\n
    WristRight\n
    HipLeft\n
    KneeLeft\n
    AnkleLeft\n
    HipRight\n
    KneeRight\n
    AnkleRight\n
    SpineShoulder\n
    """
    return ["SpineBase",
            "SpineMid",
            "Neck",
            "Head",
            "ShoulderLeft",
            "ElbowLeft",
            "WristLeft",
            "ShoulderRight",
            "ElbowRight",
            "WristRight",
            "HipLeft",
            "KneeLeft",
            "AnkleLeft",
            "HipRight",
            "KneeRight",
            "AnkleRight",
            "SpineShoulder",
            ]


def FIMitems():
    """
    0.Eating\n
    1.Grooming\n
    2.Bathing\n
    3.Dressing-Upper\n
    4.Dressing-Lower\n
    5.Toileting\n
    6.Bladder\n
    7.Bowel\n
    8.Bed_Chair_Wheelchair\n
    9.ToiletTransfer\n
    10.Tub_Shower\n
    11.Walk_Wheelchair\n
    12.Stairs\n
    Sum\n
    """
    return ["Eating",
            "Grooming",
            "Bathing",
            "Dressing-Upper",
            "Dressing-Lower",
            "Toileting",
            "Bladder",
            "Bowel",
            "Bed_Chair_Wheelchair",
            "ToiletTransfer",
            "Tub_Shower",
            "Walk_Wheelchair",
            "Stairs",
            "Sum"
            ]
