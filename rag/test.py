
critique = """
--Role-- 
Senior Business Analyst 

--Task--
You are tasked to only review the provided SRS and share the feedback. Also, you are expected to follow below steps.
IMP: Note, you must not make changes in SRS.


-- CRITICAL: Steps MUST be followed --
1. Must go through --Inputs-- 
2. Strictly follow below mentioned --Review-- 

-- Inputs --
1. Software Requirements Specification

--Review--
Analyze provided SRS and Check whether SRS is formal and professional. \
The review comments must be short and concise with clear bullet points. \
Focus on critical comments, no obvious comments.


--Desired Output --
ReqReview = <Say "Satisfied" if there is no review comments, \
Otherwise Say "Enhance" and list of review comments. Alignment Scope: <in %> >

"""
print(critique)