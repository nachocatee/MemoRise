package com.tjjhtjh.memorise.domain.tag.repository;

import com.tjjhtjh.memorise.domain.tag.repository.entity.TaggedUser;
import com.tjjhtjh.memorise.domain.tag.service.dto.response.TaggedUserResponse;

import java.util.List;

public interface TaggedUserRepositoryCustom {
    List<TaggedUser> findByTaggedUserWithMemoSeq(Long memoId);
    List<TaggedUserResponse> findByTaggedUserList(Long memoSeq);
}
