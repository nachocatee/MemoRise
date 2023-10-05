package com.tjjhtjh.memorise.domain.user.repository;

import com.tjjhtjh.memorise.domain.user.service.dto.response.UserListResponse;

import java.util.List;

public interface UserSupportRepository {
    List<UserListResponse> findByKeywordAndIsDeletedFalse(String keyword);

}
