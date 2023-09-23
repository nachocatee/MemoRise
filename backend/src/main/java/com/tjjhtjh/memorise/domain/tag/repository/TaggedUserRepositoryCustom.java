package com.tjjhtjh.memorise.domain.tag.repository;

import com.tjjhtjh.memorise.domain.tag.repository.entity.TaggedUser;

import java.util.List;

public interface TaggedUserRepositoryCustom {
    List<TaggedUser> findByTaggedUserWithMemoSeq(Long memoId);
    List<Long> findByTaggedListOfMe(Long userSeq);
}
