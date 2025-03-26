import { createClient } from '@supabase/supabase-js';
import fs from 'fs/promises';

const supabaseUrl = 'https://bljtejkgehbhbnkpphvr.supabase.co';
const supabaseKey = process.env.SUPABASE_KEY;
const supabase = createClient(supabaseUrl, supabaseKey);

const dataFile = "output_vectors.json"

async function insertCourses() {
    const fileData = await fs.readFile(dataFile, "utf-8");
    const courses = JSON.parse(fileData);
    for (const course of courses) {
        let courseObj = {
            "number": course.number,
            "metadata": course.metadata,
            "embedding": course.embedding
        };
        const { data, error } = await supabase
            .from('courses')
            .insert([courseObj]);
        if (error) {
            console.log(`🤔 Error inserting ${courseObj.number}:`, error.message);
        } else {
            continue;
        }
    }
    console.log(`✅✅✅ Inserted all courses to supabase: ${courses.length}`);
}

insertCourses()


