"""
Agronomic & Integrated Pest Management (IPM) Advisory Engine for Predict-to-Protect
Generates simple, clear, and actionable farmer-friendly preventive guidance
based on predicted pest species, outbreak risk tier, and field conditions.
"""

PEST_DETAILS = {
    'Brownplanthopper': {
        'scientific_name': 'Nilaparvata lugens',
        'common_name': 'Brown Planthopper (BPH)',
        'damage_symptom': 'Plants turn yellow, dry up in circular patches ("hopperburn"), and show black soot at the base.',
        'vulnerable_stage': 'Tillering to Panicle Initiation (Mid-growth)',
        'favorable_conditions': 'High humidity (>80%), standing water, cloudy weather, extra urea/nitrogen.',
        'high_advisory': [
            '💧 Water Drainage: Drain field water for 3 to 4 days so pests cannot multiply at the plant base.',
            '🌾 Fertilizer Pause: Stop applying extra urea (nitrogen) immediately; apply potash to harden plant stems.',
            '🔍 Plant Base Check: Inspect the bottom stem of 15–20 plants near the water line for tiny brown insects.',
            '☀️ Sunlight Paths: Create 1-foot walking alleyways every 2–3 meters to allow sunlight and airflow.',
            '🕷️ Save Friendly Spiders: Avoid harsh broad-spectrum chemical sprays so natural spiders can eat the pests.',
            '📞 Contact KVK / Officer: If you count more than 5 to 10 hoppers per plant, inform your local agriculture officer.'
        ],
        'medium_advisory': [
            '🔍 Check fields 2 times a week, looking especially at the bottom of plants.',
            '💧 Alternate wetting and drying: do not keep deep standing water continuously.',
            '🌾 Avoid heavy urea fertilizer doses.',
            '🌿 Look for early yellowing on lower leaves.'
        ],
        'low_advisory': [
            '✅ Normal Care: Continue regular weekly field checks.',
            '🌾 Balanced Fertilizer: Use recommended balanced amounts of N, P, and K.'
        ]
    },
    'Yellowstemborer': {
        'scientific_name': 'Scirpophaga incertulas',
        'common_name': 'Yellow Stem Borer (YSB)',
        'damage_symptom': 'Central plant shoot dries up ("Dead Heart") during vegetative stage, or white empty grain heads ("White Ear") appear.',
        'vulnerable_stage': 'Nursery, Tillering, and Booting stage',
        'favorable_conditions': 'Warm humid climate, cloudy days, continuous rice fields in nearby areas.',
        'high_advisory': [
            '🪤 Install Pheromone Traps: Set up 4 to 5 yellow pheromone traps per acre to catch male moths.',
            '✂️ Clip Leaf Tips: Before transplanting seedlings, cut off the top 1–2 inches of leaf tips to remove hidden insect eggs.',
            '🔍 Egg Mass Scouting: Look for yellowish-brown fuzzy egg patches on the upper surface of leaves and crush them.',
            '🌾 Regulate Urea: Avoid excess urea fertilizer; maintain adequate potash for stronger tillers.',
            '💡 Light Traps: Set up simple light traps at night to attract and monitor flying yellow moths.',
            '📞 Extension Advice: If more than 5% shoots are dried ("dead heart"), consult your local agriculture officer for bio-control.'
        ],
        'medium_advisory': [
            '🪤 Place 1–2 pheromone traps in the field to detect early moth arrival.',
            '🔍 Check 20 plants across the field for dead central shoots.',
            '🌿 Keep field borders clean of wild grassy weeds where moths hide.'
        ],
        'low_advisory': [
            '✅ Low Risk: Keep standard field surveillance and balanced crop nutrition.',
            '💡 Check nightly light trap catches once a week.'
        ]
    },
    'LeafFolder': {
        'scientific_name': 'Cnaphalocrocis medinalis',
        'common_name': 'Rice Leaf Folder',
        'damage_symptom': 'Leaves are folded lengthwise with white transparent papery streaks where caterpillars eat the green leaf tissue.',
        'vulnerable_stage': 'Tillering to Panicle Initiation',
        'favorable_conditions': 'High humidity, shady fields, heavy nitrogen use, cloudy weather.',
        'high_advisory': [
            '🎋 Rope / Branch Dragging: Pull a thorny branch or rough rope across the crop canopy early in the morning to drop caterpillars into water.',
            '🌾 Reduce Nitrogen: Stop top-dressing urea; use neem-coated urea in small split doses.',
            '🐦 Bird Perches: Place 15–20 "T"-shaped bamboo sticks per acre for insect-eating birds to sit on.',
            '🌿 Clean Field Bunds: Cut and clear wild grasses around field edges where leaf folder moths breed.',
            '🔍 Damage Threshold: If you see more than 1–2 freshly folded leaves per plant with live caterpillars inside, seek local IPM guidance.'
        ],
        'medium_advisory': [
            '🔍 Open folded leaves to check if green caterpillars or white streaks are active.',
            '🐦 Set up simple wooden perches for predatory birds.',
            '🌾 Avoid excessive urea applications.'
        ],
        'low_advisory': [
            '✅ Routine Observation: Conserve natural helpful predatory wasps and ants in the field.',
            '🌿 Maintain clean field borders.'
        ]
    },
    'Gallmidge': {
        'scientific_name': 'Orseolia oryzae',
        'common_name': 'Rice Gall Midge',
        'damage_symptom': 'Plant tillers turn into hollow silvery tubes resembling onion leaves ("Silver Shoot") with no grain formation.',
        'vulnerable_stage': 'Nursery to Early Tillering',
        'favorable_conditions': 'Continuous rain, high humidity (>85%), warm overcast days.',
        'high_advisory': [
            '🌾 Plant Resistant Varieties: For next planting, select resistant rice seeds (such as MTU 1010 or Suraksha) in midge-prone areas.',
            '💧 Water Drainage: Drain excess standing water for 2 days to stop tiny larvae from moving between tillers.',
            '🔍 Silver Shoot Check: Scout young tillers for silver onion-leaf shoots and remove heavily affected tillers.',
            '🌿 Weed Control: Remove alternate wild host grasses (like *Leersia*) from water canals and bunds.',
            '📞 Alert Officer: If silver shoots exceed 1 shoot per 20 plants, contact the nearest agriculture extension centre.'
        ],
        'medium_advisory': [
            '🔍 Inspect nurseries and young tillers closely for silver shoots.',
            '🌿 Keep irrigation canals and bunds weed-free.'
        ],
        'low_advisory': [
            '✅ Crop Health Good: Continue normal monitoring during early tillering.'
        ]
    },
    'Greenleafhopper': {
        'scientific_name': 'Nephotettix virescens',
        'common_name': 'Green Leafhopper (GLH)',
        'damage_symptom': 'Leaves turn yellow-orange at tips; hoppers spread the harmful Rice Tungro Virus disease.',
        'vulnerable_stage': 'Nursery and Vegetative stage',
        'favorable_conditions': 'Warm weather (28–32°C), high humidity, intermittent light rainfall.',
        'high_advisory': [
            '🟡 Yellow Sticky Traps: Place 4–5 yellow sticky sheets per acre to trap and monitor flying green leafhoppers.',
            '🌿 Neem Spray: Spray natural neem oil (3 to 5 ml per liter of water) or Neem Seed Kernel Extract (NSKE 5%) as a safe repellent.',
            '🔍 Check for Tungro Virus: Look for stunted plants with bright orange-yellow leaf tips and remove diseased plants immediately.',
            '🌿 Clear Field Borders: Weed field bunds thoroughly to remove wild grass shelters.',
            '💡 Light Trap Monitoring: Check night light traps for sudden influx of green hoppers.'
        ],
        'medium_advisory': [
            '🔍 Check upper leaf surfaces for small green jumping insects.',
            '💡 Use light traps at night to watch for hopper arrival.',
            '🌿 Keep field borders clean.'
        ],
        'low_advisory': [
            '✅ Normal Monitoring: Carry out routine field inspection and balanced fertilization.'
        ]
    },
    'LeafBlast': {
        'scientific_name': 'Magnaporthe oryzae',
        'common_name': 'Rice Leaf Blast (Fungus)',
        'damage_symptom': 'Diamond-shaped spots with grey ash centers and brown margins on leaves; severe cases look like burned fields.',
        'vulnerable_stage': 'Nursery and Tillering',
        'favorable_conditions': 'Night temperatures 20–24°C, high humidity (>90%), morning dew, heavy urea fertilizer.',
        'high_advisory': [
            '🌾 Stop Nitrogen Application: Completely withhold urea/nitrogen top-dressing until leaf spots stop spreading.',
            '💧 Keep Shallow Water: Maintain a steady 2-inch shallow water layer in the field to suppress fungus spore germination.',
            '🌿 Bio-Fungicide Spray: Spray *Pseudomonas fluorescens* (5 grams per liter water) or recommended bio-control formulations.',
            '🔥 Destroy Diseased Straw: Burn or deep-plow old infected crop residue so fungus cannot survive in soil.',
            '🔍 Spot Inspection: Check lower and middle leaves early in the morning during heavy dew.'
        ],
        'medium_advisory': [
            '🔍 Check leaves in early morning for new diamond-shaped spots.',
            '🌾 Avoid heavy urea fertilizer; ensure adequate potash is applied.',
            '💧 Avoid evening sprinkler irrigation that keeps leaves wet overnight.'
        ],
        'low_advisory': [
            '✅ Low Disease Pressure: Maintain recommended spacing between plants and balanced nutrition.'
        ]
    },
    'NeckBlast': {
        'scientific_name': 'Magnaporthe oryzae',
        'common_name': 'Rice Neck Blast (Fungus)',
        'damage_symptom': 'Panicle neck turns black and rots; grain heads break over and grains remain empty/chaffy.',
        'vulnerable_stage': 'Booting, Heading, and Flowering stage',
        'favorable_conditions': 'Rain during heading/flowering, prolonged morning fog/dew, cloudy days.',
        'high_advisory': [
            '🛡️ Preventive Bio-Protection: Apply protective bio-agent (*Trichoderma* or *Pseudomonas*) at 5% panicle emergence stage.',
            '🌾 Zero Nitrogen at Heading: Do not apply any urea fertilizer once panicles start emerging.',
            '💧 Good Field Drainage: Ensure good drainage so morning humidity around flower heads drops quickly.',
            '🔍 Panicle Neck Check: Inspect the junction node where the grain head emerges for dark brown or black rings.'
        ],
        'medium_advisory': [
            '🔍 Closely watch weather during flowering time; check panicle necks for dark marks.',
            '🌾 Avoid nitrogen fertilizer at panicle emergence.'
        ],
        'low_advisory': [
            '✅ Safe Condition: Normal observation during heading and grain filling.'
        ]
    },
    'Caseworm': {
        'scientific_name': 'Nymphula depunctalis',
        'common_name': 'Rice Caseworm',
        'damage_symptom': 'Leaf tips cut into small green tubes floating on standing water; leaves look white and papery.',
        'vulnerable_stage': 'Nursery to Early Tillering',
        'favorable_conditions': 'Standing stagnant water, high humidity, young tender seedlings.',
        'high_advisory': [
            '💧 Complete Water Drainage: Drain all standing water from the field for 2–3 days to dry out floating worm cases.',
            '🎋 Drag Rope Across Water: Drag a rope or thorny branch across standing water to collect floating cases at the field corner.',
            '🔍 Water Surface Check: Look for small 0.5-inch green leaf tubes floating near plant stems.',
            '🦆 Natural Predators: Encourage ducks or small fish in paddy if integrated farming is practiced.'
        ],
        'medium_advisory': [
            '🔍 Check water surface in the morning for floating leaf tubes.',
            '💧 Lower water levels to avoid stagnant deep water.'
        ],
        'low_advisory': [
            '✅ Normal Surveillance: Routine nursery and seedling check.'
        ]
    },
    'Whitebackedplanthopper': {
        'scientific_name': 'Sogatella furcifera',
        'common_name': 'White-Backed Planthopper (WBPH)',
        'damage_symptom': 'Leaves turn yellow and dry up; adults have a distinct white stripe down their back.',
        'vulnerable_stage': 'Tillering and Booting stage',
        'favorable_conditions': 'High humidity, cloudy weather, dense crop spacing, excess urea.',
        'high_advisory': [
            '💧 Water Drying: Practice alternate wetting and drying; drain field water for 3 days to lower humidity at the plant base.',
            '☀️ Sunlight Paths: Open 1-foot wide paths every 2–3 meters so sunlight reaches the lower stems.',
            '🌿 Bio-Pesticide: Spray *Beauveria bassiana* (5 grams per liter water) during high humidity morning hours.',
            '🌾 Fertilizer Pause: Stop extra nitrogen fertilizer doses; add potash to toughen stems.',
            '🕷️ Protect Friendly Bugs: Avoid harsh chemical sprays to save natural predator bugs.'
        ],
        'medium_advisory': [
            '🔍 Inspect lower leaf sheaths for white-backed hoppers.',
            '🌾 Moderate urea applications; avoid heavy single doses.'
        ],
        'low_advisory': [
            '✅ Low Risk: Continue standard weekly scouting.'
        ]
    },
    'Miridbug': {
        'scientific_name': 'Cyrtorhinus lividipennis',
        'common_name': 'Mirid Bug (Friendly Predator)',
        'damage_symptom': 'Mirid bugs are BENEFICIAL predators that eat destructive planthopper eggs and nymphs.',
        'vulnerable_stage': 'All stages',
        'favorable_conditions': 'Presence of planthoppers in the field.',
        'high_advisory': [
            '🛡️ Protect Beneficial Bugs: Do NOT spray chemical pesticides — Mirid bugs naturally destroy brown planthopper eggs.',
            '🌾 Ecological Balance: Keep border flowering plants to give shelter to beneficial predatory insects.',
            '🔍 Count Ratio: If you see 1 Mirid bug for every 5 to 10 planthoppers, natural bio-control is already actively working.'
        ],
        'medium_advisory': [
            '🔍 Observe friendly mirid bug numbers when checking for planthoppers.',
            '🛡️ Avoid unnecessary chemical spraying.'
        ],
        'low_advisory': [
            '✅ Healthy Eco-balance: Natural predatory insects are active in the field.'
        ]
    },
    'ZigZagleafhopper': {
        'scientific_name': 'Recilia dorsalis',
        'common_name': 'Zigzag Leafhopper',
        'damage_symptom': 'Wings have distinct brown zigzag "Z" marks; leafhoppers suck sap and cause leaf yellowing.',
        'vulnerable_stage': 'Tillering and Vegetative stage',
        'favorable_conditions': 'Warm humid climate, weed-covered bunds.',
        'high_advisory': [
            '🌿 Clean Canal Weeds: Remove wild grass weeds around irrigation bunds where zigzag hoppers live.',
            '🟡 Yellow Sticky Sheets: Set up yellow sticky sheets (4 per acre) to trap flying hoppers.',
            '🌿 Neem Repellent: Spray neem oil formulation (3 ml per liter water) to repel leafhoppers naturally.'
        ],
        'medium_advisory': [
            '🔍 Check field borders for zigzag-patterned hoppers.',
            '🌿 Keep bunds clean.'
        ],
        'low_advisory': [
            '✅ Standard Care: Routine crop inspection.'
        ]
    }
}

def generate_farmer_advisory(predicted_pest, risk_level, probability, farm_details, weather_details):
    """
    Generates a personalized, simple farmer-friendly advisory package.
    """
    pest_info = PEST_DETAILS.get(predicted_pest, PEST_DETAILS['Brownplanthopper'])
    
    if risk_level == 'HIGH':
        action_list = pest_info['high_advisory']
        badge_color = '#EF4444' # red
        alert_title = f"🚨 HIGH RISK ALERT: Impending {pest_info['common_name']} Outbreak"
        summary = (
            f"The AI Ensemble predicts an elevated **{probability:.1%} probability** of {pest_info['common_name']} "
            f"outbreak within the **next 2–3 weeks**. Immediate simple preventive actions are recommended below before crops are damaged."
        )
    elif risk_level == 'MEDIUM':
        action_list = pest_info['medium_advisory']
        badge_color = '#F59E0B' # amber
        alert_title = f"⚠️ MODERATE RISK WARNING: {pest_info['common_name']} Watch Needed"
        summary = (
            f"Moderate risk (**{probability:.1%}**) detected. Current humidity and weather are favorable "
            f"for pests to grow. Increase field checks and follow simple preventive steps."
        )
    else:
        action_list = pest_info['low_advisory']
        badge_color = '#10B981' # green
        alert_title = f"✅ LOW RISK STATUS: Crop Health is Safe"
        summary = (
            f"Low outbreak risk (**{probability:.1%}**). Your field conditions are safe and balanced. "
            f"Continue routine weekly crop care."
        )

    # Contextual farm suggestions
    contextual_notes = []
    growth_stage = farm_details.get('growth_stage', 'Vegetative')
    soil_type = farm_details.get('soil_type', 'Loamy')
    rainfall_trend = weather_details.get('rainfall_trend_label', 'Stable')

    if rainfall_trend == 'Increasing':
        contextual_notes.append("🌧 **Rainfall Trend Increasing**: Clear drainage outlets so water does not flood field for many days.")
    if 'Nursery' in growth_stage or 'Tillering' in growth_stage:
        contextual_notes.append("🌱 **Young Crop Stage**: Young tender tillers need extra observation for stem borers.")
    if 'Clay' in soil_type:
        contextual_notes.append("🌍 **Heavy Soil**: Soil holds moisture longer; check lower plant stems for humidity buildup.")

    return {
        'predicted_pest': predicted_pest,
        'scientific_name': pest_info['scientific_name'],
        'common_name': pest_info['common_name'],
        'risk_level': risk_level,
        'probability': probability,
        'badge_color': badge_color,
        'alert_title': alert_title,
        'summary': summary,
        'damage_symptom': pest_info['damage_symptom'],
        'vulnerable_stage': pest_info['vulnerable_stage'],
        'favorable_conditions': pest_info['favorable_conditions'],
        'action_list': action_list,
        'contextual_notes': contextual_notes
    }
