CREATE TYPE role_type AS ENUM ('Master', 'Member');

CREATE TABLE members (
    member_id SERIAL PRIMARY KEY,
    role role_type NOT NULL DEFAULT 'Member',
    fullname VARCHAR(100) NOT NULL,
    image_url TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    email varchar(255) null
);

CREATE OR REPLACE PROCEDURE add_members(
    p_role role_type,
    p_fullname varchar(100),
    p_email varchar(255),
    p_image_url text default null
)
LANGUAGE plpgsql
AS $$
BEGIN
    IF EXISTS (SELECT 1 FROM members WHERE  email = p_email) THEN
        RAISE NOTICE 'Member with the same email already exists.';
    ELSE
        INSERT INTO members (role, fullname, email, image_url)
        VALUES (p_role, p_fullname, p_email, p_image_url);
    END IF;
END;
$$;


CREATE OR REPLACE PROCEDURE update_member(
    p_member_id INT,
    p_role role_type DEFAULT NULL,
    p_fullname VARCHAR(100) DEFAULT NULL,
    p_email VARCHAR(100) DEFAULT NULL, 
    p_image_url TEXT DEFAULT NULL
)
LANGUAGE plpgsql
AS $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM members WHERE member_id = p_member_id) THEN
        RAISE NOTICE 'Member ID % does not exist', p_member_id;
    ELSE
        UPDATE members
        SET 
            role = COALESCE(p_role, role),
            fullname = COALESCE(p_fullname, fullname),
            email = COALESCE(p_email, email), 
            image_url = COALESCE(p_image_url, image_url),
            updated_at = NOW()
        WHERE member_id = p_member_id;

        RAISE NOTICE 'Member ID % updated successfully', p_member_id;
    END IF;
END;
$$;

CREATE OR REPLACE PROCEDURE delete_member(
    p_member_id INT
)
LANGUAGE plpgsql
AS $$
DECLARE
    v_max_id INT;
BEGIN
    IF NOT EXISTS (SELECT 1 FROM members WHERE member_id = p_member_id) THEN
        RAISE NOTICE 'Member ID % does not exist', p_member_id;
    ELSE
        DELETE FROM members WHERE member_id = p_member_id;
        SELECT COALESCE(MAX(member_id), 0) + 1 INTO v_max_id FROM members;
        PERFORM setval('members_member_id_seq', v_max_id, true);
        
        RAISE NOTICE 'Member ID % deleted successfully and sequence reset to %', p_member_id, v_max_id;
    END IF;
END;
$$;

CREATE OR REPLACE FUNCTION check_email_unique()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        IF NEW.email IS NOT NULL AND EXISTS (
            SELECT 1
            FROM members
            WHERE email = NEW.email
        ) THEN
            RAISE EXCEPTION 'A member with the email address % already exists.', NEW.email
            USING HINT = 'Please use a different email address.';
        END IF;
    ELSIF TG_OP = 'UPDATE' THEN
        IF NEW.email IS NOT NULL AND NEW.email <> OLD.email AND EXISTS (
            SELECT 1
            FROM members
            WHERE email = NEW.email
              AND member_id <> NEW.member_id
        ) THEN
            RAISE EXCEPTION 'A member with the email address % already exists.', NEW.email
            USING HINT = 'Please use a different email address.';
        END IF;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER check_email_unique_trigger
BEFORE INSERT OR UPDATE ON members
FOR EACH ROW EXECUTE FUNCTION check_email_unique();


---------------------------------------------
call add_members('Master','Mrs.Vo Phuong Binh','1@gmail.com');
call add_members('Member','Nguyen Thi A','2@gmail.com');
call add_members('Member','Nguyen Thi B','3@gmail.com');
call add_members('Member','Nguyen Thi C','4@gmail.com');
call add_members('Member','Nguyen Thi D','5@gmail.com');
call add_members('Member','Nguyen Thi E','6@gmail.com');

select * from members;
