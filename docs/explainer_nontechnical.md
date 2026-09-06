# Watermarking in Plain Words

**What we built: a tool that stamps an invisible ID on photos.**

## The one-liner

You can place a hidden, secret mark inside a photo. Nobody can see it. But
you can always read it back later to prove that the photo came from you.

## A little story

Imagine you make beautiful photos and you hand copies to 5 friends. Later one
of those photos shows up on a website without your permission.

- How do you know it's your photo? (Because you took it.)
- How do you prove it's yours? (Your word isn't enough.)
- Who leaked it? (All 5 friends deny it.)

With our tool you do this:

1. Before sharing, you press "embed". The photo looks **exactly the same** —
   your eyes cannot tell the difference.
2. Inside the photo is now a secret code that only you can read.
3. If the photo leaks, you press "extract". The tool scans it and reads the
   code back, telling you **which friend it originally came from**.

That's it. It's like writing your name on a banknote with invisible ink.

## Why "invisible ink" is hard

The picture is just millions of tiny colours. We tuck our secret into the
tiniest colour changes — so tiny your eye ignores them. The challenge is that
later, when the photo is saved, compressed, or edited, those tiny changes can
get damaged. Reading the secret back correctly is the hard part.

## Two kinds of reading

- **Compare-reading:** you keep the untouched original and slide it next to
  the leaked copy to spot the differences. Very reliable — but you must have
  the original.
- **Blind-reading:** you only hold the leaked photo, no original. Our
  computer program (a small "brain" trained on 800 real photos) guesses the
  hidden code from the photo alone. This is the one that matters in real
  life, because a thief never gives you the original.

## How well does it work? (the honest numbers)

- **Invisible:** the watermarked photo scores about 40 on the standard
  quality scale — that's "visually identical" to the human eye.
- **Blind-reading accuracy:** about **78 out of 100 bits** are read back
  correctly per photo.
- **Identity check:** on our test set, **96 of 100 photos** returned the
  correct hidden ID.
- **Safety:** on 100 clean photos with no mark, the tool claimed to find a
  mark **0 times** — it never raises a false alarm.

## What it does NOT yet do

- The blind-reader is still **experimental** — it doesn't beat the older
  model yet.
- Heavy editing (like strong blur) can damage the hidden mark.
- It was trained and tested on one set of photos (DIV2K), so "works on your
  exact photos" is not guaranteed.

In short: **invisible stamp + proof of ownership + way to catch the leaker —
with honest caveats.**

## Speaking with non-technical listeners

Try these one-liners:

- "It's a digital stamp you can't see, like a watermark on money."
- "Think of it as locking your initials into the photo's DNA."
- "We taught a computer to find a needle (your secret code) in a haystack
  (millions of pixels) — without being shown the original needle."

Avoid jargon in conversation. Say "colour details" instead of "DWT-SVD",
"saved/compressed" instead of "JPEG re-encode", and "false alarm" instead of
"false positive". If someone asks how it hides, say: *"We make tiny, invisible
changes to the image — every photo has millions of tiny details, so humans
never notice a few being nudged."*