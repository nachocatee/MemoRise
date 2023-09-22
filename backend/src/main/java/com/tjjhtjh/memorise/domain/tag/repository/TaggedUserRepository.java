package com.tjjhtjh.memorise.domain.tag.repository;

import com.tjjhtjh.memorise.domain.tag.repository.entity.TaggedUser;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.List;

public interface TaggedUserRepository extends JpaRepository<TaggedUser, Long> , TaggedUserRepositoryCustom {
}
