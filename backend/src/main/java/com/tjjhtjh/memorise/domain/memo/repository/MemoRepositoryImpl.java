package com.tjjhtjh.memorise.domain.memo.repository;

import com.querydsl.core.types.Projections;
import com.querydsl.jpa.impl.JPAQueryFactory;
import com.tjjhtjh.memorise.domain.memo.repository.entity.Memo;
import com.tjjhtjh.memorise.domain.memo.service.dto.response.MemoResponse;
import org.springframework.data.jpa.repository.support.QuerydslRepositorySupport;
import org.springframework.stereotype.Repository;

import java.util.ArrayList;
import java.util.List;

import static com.tjjhtjh.memorise.domain.memo.repository.entity.QMemo.memo;

@Repository
public class MemoRepositoryImpl extends QuerydslRepositorySupport implements MemoRepositoryCustom {

    private final JPAQueryFactory queryFactory;

    public MemoRepositoryImpl(JPAQueryFactory jpaQueryFactory){
        super(Memo.class);
        this.queryFactory = jpaQueryFactory;
    }

    //TODO : itemMemoList 아직 안만든 메소드임
    @Override
    public List<Memo> itemMemoList(Long itemSeq , String email) {
        List<Memo> memo = new ArrayList<>();
        return memo;
    }

    @Override
    public MemoResponse findByMemoSeq(Long memoSeq, String email) {
        return queryFactory.select(
                Projections.fields(MemoResponse.class,memo.accessType,memo.content,memo.updatedAt,memo.user.nickname))
                .from(memo).where(memo.memoSeq.eq(memoSeq)).fetchOne();
    }

}
